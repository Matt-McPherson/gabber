FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/root/.local/bin:/opt/venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y \
    git \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /opt/venv && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

COPY pyproject.toml ./
COPY uv.lock* ./
COPY README.md ./

RUN uv sync --frozen

COPY . .