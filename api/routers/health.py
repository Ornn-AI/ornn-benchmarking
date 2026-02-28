"""Health check router for the Ornn Benchmarking API."""

from __future__ import annotations

from fastapi import APIRouter

from api.config import Settings
from api.dependencies import get_app_settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness / readiness probe for Cloud Run.

    Returns service status and version information so that operators can
    verify deployment state at a glance.
    """
    settings: Settings = get_app_settings()
    return {
        "status": "ok",
        "version": settings.app_version,
        "service": settings.app_title,
    }
