FROM node:22-bookworm-slim@sha256:e9f565d32859f0cbfb52125eafe8be39390f24dac302dc3cdf566e8ba3ab21ee AS node
FROM ghcr.io/astral-sh/uv:0.13.0@sha256:cdc6093146eb3ff6a40107b38f008b789e050e77ad87865e381d9917da55a168 AS uv
FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

COPY --from=node /usr/local/ /usr/local/
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates bash libatomic1 \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global opencode-ai@1.19.4 \
    && opencode --version

WORKDIR /opt/generator
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs/webgen-generation.yaml configs/webgen-builder.md ./configs/
RUN uv sync --locked --no-dev \
    && useradd --create-home --uid 1000 generator
ARG SOURCE_COMMIT
RUN test -n "$SOURCE_COMMIT" && printf '%s\n' "$SOURCE_COMMIT" > source-commit.txt
ENV PATH="/opt/generator/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GENERATION_CONTAINER=1
USER generator
ENTRYPOINT ["python", "-m", "decomp_judge.webgen_generate"]
