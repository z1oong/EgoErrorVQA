FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim
RUN adduser --disabled-password agentbeats
USER agentbeats
WORKDIR /EgoErrorVQA
COPY --chown=agentbeats pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/home/agentbeats/.cache/uv,uid=1000 uv sync --frozen --no-dev
COPY --chown=agentbeats src src
ENTRYPOINT ["uv", "run", "python", "main.py", "launch"]