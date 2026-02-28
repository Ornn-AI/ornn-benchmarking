"""API key authentication for the Ornn Benchmarking API.

Provides a FastAPI dependency that validates the ``X-API-Key`` header
against a set of known valid keys.  Revoked keys are tracked separately
so they can be rejected with a clear 401 rather than treated as unknown.

Keys are loaded from environment variables for simplicity.  In production,
this could be backed by Firestore or Secret Manager.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status

# ---------------------------------------------------------------------------
# Key management helpers
# ---------------------------------------------------------------------------

_VALID_API_KEYS: set[str] | None = None
_REVOKED_API_KEYS: set[str] | None = None


def _load_valid_keys() -> set[str]:
    """Load valid API keys from the ``ORNN_API_KEYS`` environment variable.

    Keys are comma-separated.  Falls back to a development-only default
    when the variable is unset (to ease local testing).
    """
    global _VALID_API_KEYS
    if _VALID_API_KEYS is None:
        raw = os.environ.get("ORNN_API_KEYS", "dev-test-key")
        _VALID_API_KEYS = {k.strip() for k in raw.split(",") if k.strip()}
    return _VALID_API_KEYS


def _load_revoked_keys() -> set[str]:
    """Load revoked API keys from the ``ORNN_REVOKED_API_KEYS`` env var."""
    global _REVOKED_API_KEYS
    if _REVOKED_API_KEYS is None:
        raw = os.environ.get("ORNN_REVOKED_API_KEYS", "")
        _REVOKED_API_KEYS = {k.strip() for k in raw.split(",") if k.strip()}
    return _REVOKED_API_KEYS


def reset_api_keys() -> None:
    """Reset cached key sets (used by tests)."""
    global _VALID_API_KEYS, _REVOKED_API_KEYS
    _VALID_API_KEYS = None
    _REVOKED_API_KEYS = None


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------


async def require_api_key(
    x_api_key: str | None = Header(default=None),
) -> str:
    """Validate the ``X-API-Key`` header and return the key on success.

    Raises :class:`HTTPException` (401) when the key is missing, invalid,
    or revoked.  Response bodies never expose which keys exist or why
    a specific key was rejected — just a generic ``Unauthorized`` message.
    """
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide a valid key via the X-API-Key header.",
        )

    revoked = _load_revoked_keys()
    if x_api_key in revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized.",
        )

    valid = _load_valid_keys()
    if x_api_key not in valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized.",
        )

    return x_api_key
