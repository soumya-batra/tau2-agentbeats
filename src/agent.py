import json
import logging
import time
import uuid
from typing import Any, List, Optional

import asyncio
import nest_asyncio
from pydantic import BaseModel, HttpUrl, ValidationError

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from tau2.agent.base import BaseAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgentState
from tau2.data_model.message import (
    AssistantMessage, MultiToolMessage, SystemMessage,
    ToolCall, ToolMessage, UserMessage
)
from tau2.environment.tool import Tool
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.user.user_simulator import UserSimulator
from tau2.evaluator.evaluator import evaluate_simulation, EvaluationType

from messenger import Messenger

nest_asyncio.apply()

RESPOND_ACTION_NAME = "respond"


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


def tools_to_str(tools: List[Tool]) -> str:
    return json.dumps([tool.openai_schema for tool in tools], indent=2)


def extract_text_from_message(message: MultiToolMessage | UserMessage | ToolMessage) -> str | None:
    if isinstance(message, UserMessage):
        return message.content
    elif isinstance(message, MultiToolMessage):
        tool_results = []
        for tm in message.tool_messages:
            tool_results.append(f"Tool '{tm.name}' result: {tm.content}")
        return "\n".join(tool_results)
    else:
        return str(message.content) if hasattr(message, 'content') else str(message)


def get_task_objects(domain: str, task_ids: Optional[List[str]], num_tasks: Optional[int] = None):
    task_set_name = domain
    task_split_name = "base"
    if task_ids is None:
        tasks = get_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    else:
        tasks = get_tasks(
            task_set_name=task_set_name,
            task_split_name=task_split_name,
            task_ids=task_ids,
        )
    if num_tasks is not None:
        tasks = tasks[:num_tasks]
    return tasks


class RemoteA2AAgent(BaseAgent):
    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        messenger: Messenger,
        agent_url: str,
    ):
        self.tools = tools
        self.domain_policy = domain_policy
        self.messenger = messenger
        self.agent_url = agent_url
        self._is_first_message = True

    @property
    def agent_prompt(self) -> str:
        return f"""{self.domain_policy}

Here's a list of tools you can use (you can use at most one tool at a time):
{tools_to_str(self.tools)}

Additionally, you can respond to the user with the following call:

{json.dumps({
    "type": "function",
    "function": {
        "name": RESPOND_ACTION_NAME,
        "description": "Respond directly to the user with a message instead of calling a tool.",
        "parameters": {
            "properties": {
                "content": {
                    "description": "The message content to send to the user.",
                    "title": "Content",
                    "type": "string"
                }
            },
            "required": ["content"],
            "title": "parameters",
            "type": "object"
        }
    }
}, indent=2)}


Please respond in JSON format.
The JSON should contain:
- "name": the tool call function name.
- "arguments": the arguments for the tool call.

You should only use one tool at a time!
You cannot respond to user and use a tool at the same time!
Tool calls are cheap and you should not hesitate to use them when necessary.
Most tasks will require you to use tools and respond to the user as part of the optimal solution.

Examples of responses:
{json.dumps({"name": "find_user_id_by_name_zip", "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip_code": "19122"}}, indent=2)}

{json.dumps({"name": RESPOND_ACTION_NAME, "arguments": {"content": "Hello, how can I help you today?"}}, indent=2)}
"""

    def get_init_state(self, message_history: Optional[list] = None) -> LLMAgentState:
        if message_history is None:
            message_history = []
        self._is_first_message = True
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.agent_prompt)],
            messages=message_history,
        )

    def set_seed(self, seed: int):
        pass

    def stop(self, last_message=None, state=None):
        pass

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        outgoing_text = extract_text_from_message(message)

        if self._is_first_message:
            outgoing_text = f"{self.agent_prompt}\n\nNow here are the user messages:\n{'\n'.join([extract_text_from_message(msg) for msg in state.messages])}"

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        response = loop.run_until_complete(
            self.messenger.talk_to_agent(
                message=outgoing_text,
                url=self.agent_url,
                new_conversation=self._is_first_message,
            )
        )
        self._is_first_message = False

        assistant_message = self._parse_response(response)
        state.messages.append(assistant_message)

        return assistant_message, state

    def _parse_response(self, response: str) -> AssistantMessage:
        try:
            parsed = json.loads(response)

            if isinstance(parsed, list):
                if len(parsed) == 0:
                    raise ValueError("Empty list returned")
                action_dict = parsed[0]
            elif isinstance(parsed, dict):
                action_dict = parsed
            else:
                raise ValueError(f"Unexpected JSON type: {type(parsed)}")

            is_tool_call = action_dict["name"] != RESPOND_ACTION_NAME

            if not is_tool_call:
                return AssistantMessage(
                    role="assistant",
                    content=action_dict["arguments"]["content"],
                    tool_calls=None,
                )
            else:
                tool_call = ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=action_dict["name"],
                    arguments=action_dict["arguments"],
                    requestor="assistant",
                )
                return AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            logging.warning(f"Failed to parse agent response as JSON: {e}")
            logging.warning(f"Response was: {response[:200]}")
            return AssistantMessage(
                role="assistant",
                content=response,
                tool_calls=None,
            )


class Agent:
    required_roles = ["agent"]
    required_config_keys = ["domain"]

    def __init__(self):
        self.messenger = Messenger()

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        domain = request.config["domain"]
        task_ids = request.config.get("task_ids", None)
        num_tasks = request.config.get("num_tasks", None)
        max_steps = request.config.get("max_steps", 200)
        user_llm = request.config.get("user_llm", "openai/gpt-4o-mini")
        user_llm_args = request.config.get("user_llm_args", {"temperature": 1.0})
        agent_url = str(request.participants["agent"])

        tasks = get_task_objects(domain, task_ids, num_tasks)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting evaluation of {len(tasks)} tasks in {domain} domain")
        )

        metrics = {"tasks": {}}
        start_time = time.time()

        try:
            for task in tasks:
                task_id = task.id
                logging.info(f"Running task {task_id}...")
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(f"Running task {task_id}...")
                )

                try:
                    reward = await self._run_single_task(
                        agent_url=agent_url,
                        domain=domain,
                        task=task,
                        max_steps=max_steps,
                        user_llm=user_llm,
                        user_llm_args=user_llm_args,
                    )
                    metrics["tasks"][task_id] = reward
                    logging.info(f"Task {task_id} completed with reward: {reward}")
                except Exception as e:
                    logging.error(f"Task {task_id} failed: {e}", exc_info=True)
                    metrics["tasks"][task_id] = 0.0

            time_used = time.time() - start_time
            total_reward = sum(metrics["tasks"].values())
            num_completed = len(metrics["tasks"])
            pass_rate = (total_reward / num_completed * 100) if num_completed > 0 else 0

            result_data = {
                "domain": domain,
                "score": total_reward,
                "max_score": num_completed,
                "pass_rate": pass_rate,
                "task_rewards": metrics["tasks"],
                "time_used": time_used,
            }

            task_results_str = "\n".join(
                f"  {task_id}: {'✓' if reward == 1.0 else '✗'} ({reward})"
                for task_id, reward in metrics["tasks"].items()
            )

            summary = f"""Tau2 Benchmark Results
Domain: {domain}
Tasks: {num_completed}
Pass Rate: {pass_rate:.1f}% ({int(total_reward)}/{num_completed})
Time: {time_used:.1f}s

Task Results:
{task_results_str}"""

            await updater.add_artifact(
                parts=[
                    Part(root=TextPart(text=summary)),
                    Part(root=DataPart(data=result_data)),
                ],
                name="Result",
            )

        finally:
            self.messenger.reset()

    async def _run_single_task(
        self,
        agent_url: str,
        domain: str,
        task,
        max_steps: int,
        user_llm: str,
        user_llm_args: dict,
    ) -> float:
        env_constructor = registry.get_env_constructor(domain)
        environment = env_constructor(solo_mode=False)

        agent = RemoteA2AAgent(
            tools=environment.get_tools(),
            domain_policy=environment.get_policy(),
            messenger=self.messenger,
            agent_url=agent_url,
        )

        user = UserSimulator(
            tools=environment.get_user_tools() if environment.user_tools else None,
            instructions=str(task.user_scenario),
            llm=user_llm,
            llm_args=user_llm_args,
        )

        orchestrator = Orchestrator(
            domain=domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            max_errors=10,
            seed=42,
            solo_mode=False,
            validate_communication=False,
        )

        simulation_run = orchestrator.run()

        logging.info(f"Task {task.id} terminated: {simulation_run.termination_reason}")

        try:
            reward_info = evaluate_simulation(
                simulation=simulation_run,
                task=task,
                evaluation_type=EvaluationType.ACTION,
                solo_mode=False,
                domain=domain,
            )
            return reward_info.reward
        except Exception as e:
            logging.error(f"Evaluation failed for task {task.id}: {e}")
            return 0.0
