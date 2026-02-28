#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

./.venv/bin/python -m pip install --upgrade pip setuptools wheel

if [ -f "pyproject.toml" ]; then
  ./.venv/bin/python -m pip install -e ".[dev]"
else
  ./.venv/bin/python -m pip install \
    typer \
    rich \
    pydantic \
    fastapi \
    uvicorn \
    google-cloud-firestore \
    httpx \
    pytest \
    pytest-asyncio \
    ruff \
    mypy \
    build
fi
