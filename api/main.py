"""FastAPI application for the Ornn Benchmarking API."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="Ornn Benchmarking API",
    description="API for receiving, storing, and verifying GPU benchmark results.",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
