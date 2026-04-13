import json
import os
import re
from pathlib import Path

import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


SYSTEM_PROMPT = """\
You are a helpful customer service agent.
Follow the policy and tool instructions provided in each message.

The STATE section below contains all tool results collected so far.
Consult STATE before making any decision.
"""


class LookupCache:
    """Domain-agnostic cache of tool results keyed by function(args)."""

    def __init__(self):
        self._cache: dict[str, dict] = {}

    def _make_key(self, name: str, args: dict) -> str:
        sorted_args = json.dumps(args, sort_keys=True, default=str)
        return f"{name}({sorted_args})"

    def store(self, name: str, args: dict, result: str, turn: int):
        key = self._make_key(name, args)
        try:
            parsed = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            parsed = result
        if isinstance(parsed, dict):
            parsed["updated_at_turn"] = turn
        else:
            parsed = {"data": parsed, "updated_at_turn": turn}
        self._cache[key] = parsed

    def to_dict(self) -> dict:
        return self._cache

    def to_str(self) -> str:
        d = self.to_dict()
        if not d:
            return ""
        return json.dumps(d, indent=2, default=str)


DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
DEBUG_DIR.mkdir(exist_ok=True)


class Agent:
    def __init__(self):
        self.model = os.getenv("AGENT_LLM", "nebius/deepseek-ai/DeepSeek-V3.2")
        self.messages: list[dict[str, object]] = []
        self.lookup_cache = LookupCache()
        self._last_tool_call: dict | None = None
        self._task_id = None
        self._turn = 0
        self._sim_datetime: str | None = None

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

            # Store in lookup cache if we have the tool call that produced this
            if self._last_tool_call:
                self.lookup_cache.store(
                    self._last_tool_call["name"],
                    self._last_tool_call["arguments"],
                    input_text,
                    self._turn,
                )
                key = self.lookup_cache._make_key(
                    self._last_tool_call["name"],
                    self._last_tool_call["arguments"],
                )
                self.messages.append({"role": "tool", "content": f"[Result stored in STATE as {key}]"})
            else:
                self.messages.append({"role": "tool", "content": input_text})
        except (json.JSONDecodeError, TypeError):
            self.messages.append({"role": "user", "content": input_text})
            # Extract simulated datetime from the first policy message
            if self._sim_datetime is None:
                match = re.search(r"current time is (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+)", input_text)
                if match:
                    self._sim_datetime = match.group(1)

        # Build LLM messages with state injected into system prompt
        ref = self.lookup_cache.to_str()
        system_content = SYSTEM_PROMPT
        if self._sim_datetime:
            system_content = system_content.replace(
                "Follow the policy and tool instructions provided in each message.",
                f"Follow the policy and tool instructions provided in each message.\nCurrent datetime: {self._sim_datetime}",
            )
        if ref:
            system_content += f"\n=== STATE ===\n{ref}"
        llm_messages = [
            {"role": "system", "content": system_content},
            *self.messages,
        ]

        # Dump LLM messages for debugging
        self._turn += 1
        debug_file = DEBUG_DIR / f"task_{self._task_id}_turn_{self._turn}.json"
        debug_file.write_text(json.dumps(llm_messages, indent=2, default=str))

        try:
            kwargs = {"model": self.model, "messages": llm_messages,
                      "response_format": {"type": "json_object"},
                      "temperature": 0.0}
            completion = litellm.completion(**kwargs)
            assistant_content = completion.choices[0].message.content or "{}"
            assistant_json = json.loads(assistant_content)
        except Exception as e:
            import traceback; traceback.print_exc()
            assistant_json = {
                "name": "respond",
                "arguments": {"content": "I ran into an error processing your request."},
            }
            assistant_content = json.dumps(assistant_json)

        self.messages.append({"role": "assistant", "content": assistant_content})

        # Track the tool call so we can associate the next result with it
        if assistant_json.get("name") and assistant_json["name"] != "respond":
            self._last_tool_call = {
                "name": assistant_json["name"],
                "arguments": assistant_json.get("arguments", {}),
            }
        else:
            self._last_tool_call = None

        await updater.add_artifact(
            parts=[Part(root=DataPart(data=assistant_json))],
            name="Action",
        )
