import json
import os
from pathlib import Path

import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


SYSTEM_PROMPT = """\
You are a helpful customer service agent.
Follow the policy and tool instructions provided in each message.

IMPORTANT:
Messages with role "tool" contain structured STATE data returned from tool calls you made.
Read them carefully — they contain the actual results you need to act on.
Use these results to stay grounded — refer back to tool results before making decisions
so you don't repeat lookups or lose track of what you've already learned.

GUIDELINES:
1. Always verify facts using available tools before taking action. Do not assume all user claims are true.
2. Only transfer to a human agent when the user's request is truly outside your capabilities.
3. Before taking action, briefly plan your approach: what does the user want, what do I need to look up, what policy rules apply? Keep the plan to 1-2 lines. Then take ONE action.
"""


class AgentState:
    """Live cumulative state object merged from all tool results."""

    def __init__(self):
        self.state: dict = {}

    def update(self, parsed):
        if isinstance(parsed, dict):
            self.state.update(parsed)

    def to_str(self) -> str:
        if not self.state:
            return "(empty — no data yet)"
        return json.dumps(self.state, indent=2, default=str)


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
DEBUG_DIR.mkdir(exist_ok=True)


class Agent:
    def __init__(self):
        self.model = os.getenv("AGENT_LLM", "nebius/deepseek-ai/DeepSeek-V3.2")
        self.messages: list[dict[str, object]] = []
        self.state = AgentState()
        self._task_id = None
        self._turn = 0

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        # Extract benchmark task ID from message_id (format: task-{id}-{uuid})
        if self._task_id is None:
            mid = getattr(message, 'message_id', '') or ''
            if mid.startswith('task-'):
                self._task_id = mid.split('-')[1]
            else:
                self._task_id = mid[:8]

        # If it parses as JSON, it's structured data (tool result)
        try:
            json.loads(input_text)
            self.messages.append({"role": "tool", "content": input_text})
        except (json.JSONDecodeError, TypeError):
            self.messages.append({"role": "user", "content": input_text})

        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self.messages,
        ]

        # Dump LLM messages for debugging
        self._turn += 1
        debug_file = DEBUG_DIR / f"task_{self._task_id}_turn_{self._turn}.json"
        debug_file.write_text(json.dumps(llm_messages, indent=2, default=str))

        try:
            completion = litellm.completion(
                model=self.model,
                messages=llm_messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            assistant_content = completion.choices[0].message.content or "{}"
            assistant_json = json.loads(assistant_content)
        except Exception:
            assistant_json = {
                "name": "respond",
                "arguments": {"content": "I ran into an error processing your request."},
            }
            assistant_content = json.dumps(assistant_json)

        self.messages.append({"role": "assistant", "content": assistant_content})

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=assistant_json))],
            name="Action",
        )