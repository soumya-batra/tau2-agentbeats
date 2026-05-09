# Tau2 Purple Agent

A customer service agent built for the [Tau2 benchmark](https://github.com/sierra-research/tau2-bench) on the [AgentBeats](https://agentbeats.dev) platform. Communicates via the [A2A protocol](https://a2a-protocol.org/latest/) and is evaluated on multi-turn customer service tasks across airline, retail, and telecom domains.

## Results

Base Model useed: Qwen3.5-397B-A17B

| Domain | Score |
|--------|-------|
| Telecom | 101/114 |
| Retail | 55/104 |
| Airline | 42/50 |

## Approach: Plan-Reason-Verify-Act Loop

We implement a **Plan → Reason → Verify → Act** loop. In the first turn, the agent creates a short plan with the steps and immediate next step, and keeps updating it as new information arrives. Based on this plan, it reasons about its current action. We also apply self-critique, inspired by promising research in this direction, where the agent verifies its reasoning against the goal it is trying to achieve. If verification fails, a revised reasoning and verification step is generated until validation passes. This happens within a single LLM call, avoiding additional round trips.

Each JSON response contains:
1. **Plan** — working memory with goal, steps, gathered facts, completed steps, and next step
2. **Reason** — why the chosen action is the right next step given the current plan and facts
3. **Verify** — self-critique checking whether the action advances the goal and follows policy correctly
4. **Act** — the tool call or user response to execute

The plan is a structured JSON object that persists across turns:
```json
{
  "plan": {
    "goal": "what the user wants",
    "steps": ["list of steps to execute"],
    "state": {"key facts gathered so far as key-value pairs"},
    "done": ["completed steps"],
    "next": "immediate next step"
  },
  "reason": "why this action is the right next step",
  "verify": "self-critique: does this action advance the goal and follow policy correctly?",
  "name": "tool_name_or_respond",
  "arguments": {}
}
```

## Project Structure

```
src/
├─ server.py      # Server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Agent implementation (Plan-Reason-Verify-Act loop)
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # Agent tests
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
amber-manifest.json5  # Amber manifest
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

## Running with Docker

```bash
# Build the image
docker build -t tau2-purple .

# Run the container
docker run -p 9019:9019 tau2-purple
```
