# Dockerfile for Ornn Benchmarking API — Cloud Run deployment
#
# This image runs the FastAPI backend only (not the CLI benchmark runner).
# Designed for Cloud Run with scale-to-zero free-tier-safe defaults.
#
# Build:
#   docker build -t ornn-api .
#
# Run locally:
#   docker run -p 8080:8080 -e PORT=8080 ornn-api

FROM python:3.12-slim AS base

# Prevent Python from writing bytecode and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy package metadata and source needed by setuptools to resolve ".[api]"
COPY pyproject.toml ./
COPY src/ ./src/

# Install only API dependencies (server-side only, no benchmark tools)
RUN pip install --no-cache-dir ".[api]"

# Copy API source code
COPY api/ ./api/

# Cloud Run injects PORT env var; default to 8080
ENV PORT=8080

EXPOSE ${PORT}

# Run with uvicorn; Cloud Run sends SIGTERM for graceful shutdown
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
