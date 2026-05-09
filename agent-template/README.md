# Tau2 Purple Agent

A purple agent for the [Tau2 benchmark](https://github.com/sierra-research/tau2-bench) on the [AgentBeats](https://agentbeats.dev) platform. Solves customer service tasks across multiple domains (airline, retail, telecom) via the [A2A protocol](https://a2a-protocol.org/latest/).

## Approach: Plan-Reason-Verify-Act Loop

We implement a **Plan → Reason → Verify → Act** loop. In the first turn, the agent creates a short plan with steps and an immediate next step, and keeps updating it as new information arrives. Based on this plan, it reasons about its current action. We also apply self-critique, inspired by promising research in this direction, where the agent verifies its reasoning against the goal it is trying to achieve. If verification fails, a revised reasoning and verification step is generated until validation passes. This happens within a single LLM call, avoiding additional round trips.

Each JSON response contains:
1. **Plan** — working memory with goal, steps, gathered facts, completed steps, and next step
2. **Reason** — why the chosen action is the right next step given the current plan and facts
3. **Verify** — self-critique checking whether the action advances the goal and follows policy correctly
4. **Act** — the tool call or user response to execute

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
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
