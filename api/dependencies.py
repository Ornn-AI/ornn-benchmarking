"""Dependency injection for the Ornn Benchmarking API.

Provides injectable abstractions (Firestore client, settings, rate limiter)
that can be overridden in tests without touching production wiring.
"""

from __future__ import annotations

from typing import Any, Protocol

from api.config import Settings, get_settings
from api.rate_limit import RateLimiter

# ---------------------------------------------------------------------------
# Firestore client protocol - enables mock / emulator substitution in tests
# ---------------------------------------------------------------------------


class FirestoreClientProtocol(Protocol):
    """Minimal interface for Firestore operations used by the API.

    Production code uses the real ``google.cloud.firestore.Client``; tests can
    supply a lightweight fake that satisfies this protocol.
    """

    def collection(self, path: str) -> Any:
        """Return a collection reference."""
        ...


# ---------------------------------------------------------------------------
# Singleton holders (set once at app startup, swapped in tests)
# ---------------------------------------------------------------------------

_settings: Settings | None = None
_firestore_client: FirestoreClientProtocol | None = None
_rate_limiter: RateLimiter | None = None


def get_app_settings() -> Settings:
    """Return the cached application settings singleton."""
    global _settings
    if _settings is None:
        _settings = get_settings()
    return _settings


def set_app_settings(settings: Settings) -> None:
    """Override application settings (used by tests)."""
    global _settings
    _settings = settings


def get_firestore_client() -> FirestoreClientProtocol:
    """Return the Firestore client, creating it lazily on first access.

    In production the real ``google.cloud.firestore.Client`` is instantiated
    once and reused.  Tests call :func:`set_firestore_client` to inject a fake
    *before* any endpoint handler runs.
    """
    global _firestore_client
    if _firestore_client is None:
        from google.cloud import firestore

        settings = get_app_settings()
        _firestore_client = firestore.Client(project=settings.firestore_project_id)
    return _firestore_client


def set_firestore_client(client: FirestoreClientProtocol) -> None:
    """Override the Firestore client (used by tests to inject a fake)."""
    global _firestore_client
    _firestore_client = client


def get_rate_limiter() -> RateLimiter:
    """Return the rate limiter, creating it lazily on first access.

    Constructs the limiter from :func:`get_app_settings` values.
    Tests call :func:`set_rate_limiter` to inject a custom instance.
    """
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_app_settings()
        _rate_limiter = RateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Override the rate limiter (used by tests)."""
    global _rate_limiter
    _rate_limiter = limiter


def reset_dependencies() -> None:
    """Reset all cached singletons (used between tests)."""
    global _settings, _firestore_client, _rate_limiter
    _settings = None
    _firestore_client = None
    if _rate_limiter is not None:
        _rate_limiter.reset()
    _rate_limiter = None
