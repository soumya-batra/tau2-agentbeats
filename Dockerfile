FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

COPY pyproject.toml uv.lock README.md ./
COPY src src

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

RUN git clone --depth 1 https://github.com/sierra-research/tau2-bench.git tau2-bench

ENV TAU2_DATA_DIR=/home/agent/tau2-bench/data

ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0"]
EXPOSE 8081
