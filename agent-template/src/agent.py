import json
import os
from pathlib import Path

import litellm

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Message, Part, TaskState
from a2a.utils import get_message_text, new_agent_text_message


SYSTEM_PROMPT = """\
You are a customer service agent that helps the user according to the <policy> using the tools provided below, and your <current_plan>.
Follow the policy and use the tools available to you, in each message.

IMPORTANT:
Messages with role "tool" contain structured STATE data returned from tool calls you made.
Read them carefully — they contain the actual results you need to act on.
Use these results to stay grounded — refer back to tool results before making decisions
so you don't repeat lookups or lose track of what you've already learned.

If information is missing, ask the user; do not invent slot values.

{tool_instructions}

Use a Plan-Reason-Verify-Act loop. In every JSON response, include these fields:

1. "plan": Update your working memory — track your goal, steps, gathered facts, and next step.
2. "reason": Explain WHY you are choosing this specific action right now, based on your plan and the information gathered so far.
3. "verify": Self-critique your reasoning — does this action actually advance your goal? Are you following the policy correctly? Are there any facts you are assuming that you haven't confirmed? If verification fails, revise your reason and choose a different action.
4. "name" and "arguments": The action to execute.

Respond with valid JSON in this format:
{{"plan": {{"goal": "what the user wants", "steps": ["list of steps to execute"], "state": {{"key facts gathered so far as key-value pairs"}}, "done": ["completed steps"], "next": "immediate next step"}}, "reason": "why this action is the right next step", "verify": "self-critique: does this action advance the goal and follow policy correctly?", "name": "tool_name_or_respond", "arguments": {{}}}}

<policy>
{policy}
</policy>

<current_plan>
{plan_content}
</current_plan>"""

DEBUG_DIR = Path(__file__).resolve().parent.parent / "debug"
DEBUG_DIR.mkdir(exist_ok=True)


def parse_first_message(text):
    """Parse the green agent's first message into policy, tool instructions, and user text."""
    tools_header = "Here's a list of tools you can use"
    user_msgs_header = "Now here are the user messages:"

    tools_header_pos = text.find(tools_header)
    user_msgs_pos = text.find(user_msgs_header)

    # Extract policy (everything before tools header)
    policy = text[:tools_header_pos].strip() if tools_header_pos > 0 else ""

    # Extract full tool instructions block (tools + respond schema + format + examples)
    tool_instructions = ""
    if tools_header_pos > 0 and user_msgs_pos > 0:
        tool_instructions = text[tools_header_pos:user_msgs_pos].strip()

    # Extract user messages (after the header)
    user_text = ""
    if user_msgs_pos > 0:
        user_text = text[user_msgs_pos + len(user_msgs_header) :].strip()

    return policy, tool_instructions, user_text


class Agent:
    def __init__(self):
        self.model = os.getenv("AGENT_LLM", "nebius/Qwen/Qwen3.5-397B-A17B-fast")
        self.messages: list[dict[str, object]] = []
        self._task_id = None
        self._turn = 0
        self._tool_instructions = ""
        self._policy = ""
        self._initialized = False
        self._plan = ""

    def _init_from_first_message(self, text):
        """Parse tool instructions and policy from the green agent's first message."""
        self._policy, self._tool_instructions, user_text = parse_first_message(text)
        self._initialized = True
        # First line is the agent greeting, rest are user messages
        if user_text:
            first = True
            for line in user_text.split("\n"):
                line = line.strip()
                if line:
                    if first:
                        self.messages.append({"role": "assistant", "content": line})
                        first = False
                    else:
                        self.messages.append({"role": "user", "content": line})

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        await updater.update_status(TaskState.working, new_agent_text_message("Thinking..."))

        # Extract benchmark task ID
        if self._task_id is None:
            mid = getattr(message, "message_id", "") or ""
            if mid.startswith("task-"):
                self._task_id = mid.split("-")[1]
            else:
                self._task_id = mid[:8]

        # First message: parse tools and policy from green agent's prompt
        if not self._initialized:
            self._init_from_first_message(input_text)
        else:
            self.messages.append({"role": "user", "content": input_text})

        # Build system prompt
        if self._plan:
            plan_content = self._plan + "\nUpdate this plan as you make progress."
        else:
            plan_content = "No plan yet. Build one from the user's request."

        system_content = (
            SYSTEM_PROMPT
            .replace("{policy}", self._policy)
            .replace("{plan_content}", plan_content)
            .replace("{tool_instructions}", self._tool_instructions)
        )

        llm_messages = [
            {"role": "system", "content": system_content},
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
                response_format={"type": "json_object"},
                temperature=0.6,
                top_p=0.95,
                extra_body={"top_k": 20, "min_p": 0},
            )
            msg_obj = completion.choices[0].message
            reasoning = getattr(msg_obj, "reasoning_content", None)
            raw_content = msg_obj.content or "{}"

            # Parse JSON response
            parsed = json.loads(raw_content)

            # Extract and store plan
            plan = parsed.pop("plan", None)
            if plan:
                self._plan = json.dumps(plan)

            # Extract action (name + arguments)
            func_name = parsed.get("name", "respond")
            func_args = parsed.get("arguments", {})
            assistant_json = {"name": func_name, "arguments": func_args}

            # Store the raw JSON as assistant message
            self.messages.append({"role": "assistant", "content": raw_content})

        except Exception:
            assistant_json = {
                "name": "respond",
                "arguments": {"content": "I ran into an error processing your request."},
            }
            reasoning = None

        # Save full response with reasoning to debug
        debug_response = DEBUG_DIR / f"task_{self._task_id}_turn_{self._turn}_response.json"
        debug_response.write_text(
            json.dumps(
                {
                    "content": assistant_json,
                    "reasoning_content": reasoning,
                    "plan": self._plan,
                },
                indent=2,
                default=str,
            )
        )

        # Send action to green agent
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=assistant_json))],
            name="Action",
        )
