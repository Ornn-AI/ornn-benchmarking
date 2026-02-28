"""Application settings for the Ornn Benchmarking API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Immutable configuration for the Ornn Benchmarking API.

    Reads from environment variables with sensible defaults for local development.
    """

    # GCP / Firestore
    firestore_project_id: str = field(
        default_factory=lambda: os.environ.get("FIRESTORE_PROJECT_ID", "ornn-benchmarking")
    )
    firestore_emulator_host: str | None = field(
        default_factory=lambda: os.environ.get("FIRESTORE_EMULATOR_HOST")
    )

    # API metadata
    api_version: str = "v1"
    app_title: str = "Ornn Benchmarking API"
    app_description: str = (
        "API for receiving, storing, and verifying GPU benchmark results."
    )
    app_version: str = "0.1.0"

    # Rate limiting (per API key)
    rate_limit_requests: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_REQUESTS", "60"))
    )
    rate_limit_window_seconds: int = field(
        default_factory=lambda: int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))
    )

    # Server
    port: int = field(
        default_factory=lambda: int(os.environ.get("PORT", "8080"))
    )
    debug: bool = field(
        default_factory=lambda: os.environ.get("DEBUG", "").lower() in ("1", "true", "yes")
    )


def get_settings() -> Settings:
    """Return application settings (constructed from environment)."""
    return Settings()
