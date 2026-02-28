"""Dependency injection for the Ornn Benchmarking API.

Provides injectable abstractions (Firestore client, settings) that can be
overridden in tests without touching production wiring.
"""

from __future__ import annotations

from typing import Any, Protocol

from api.config import Settings, get_settings

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


def reset_dependencies() -> None:
    """Reset all cached singletons (used between tests)."""
    global _settings, _firestore_client
    _settings = None
    _firestore_client = None
