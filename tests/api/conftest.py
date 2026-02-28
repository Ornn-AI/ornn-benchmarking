"""Shared fixtures for API tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from typing import Any

import pytest
from api.auth import reset_api_keys
from api.dependencies import reset_dependencies, set_firestore_client, set_rate_limiter
from api.rate_limit import RateLimiter
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Lightweight Firestore fake for unit tests
# ---------------------------------------------------------------------------


class FakeDocumentRef:
    """Minimal fake Firestore document reference."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self) -> FakeDocumentRef:
        return self

    @property
    def exists(self) -> bool:
        return bool(self._data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeQueryResult:
    """Minimal fake for a Firestore query result document."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeQuery:
    """Minimal fake for a Firestore query (``where(...).limit(...)``)."""

    def __init__(self, docs: list[FakeDocumentRef], field: str, value: Any) -> None:
        self._docs = docs
        self._field = field
        self._value = value
        self._limit: int | None = None

    def limit(self, n: int) -> FakeQuery:
        self._limit = n
        return self

    def stream(self) -> list[FakeQueryResult]:
        results: list[FakeQueryResult] = []
        for doc in self._docs:
            if doc.exists and doc.to_dict().get(self._field) == self._value:
                results.append(FakeQueryResult(doc.to_dict()))
                if self._limit is not None and len(results) >= self._limit:
                    break
        return results


class FakeCollectionRef:
    """Minimal fake Firestore collection reference."""

    def __init__(self) -> None:
        self._docs: dict[str, FakeDocumentRef] = {}

    def document(self, doc_id: str | None = None) -> FakeDocumentRef:
        if doc_id is None:
            doc_id = uuid.uuid4().hex
        if doc_id not in self._docs:
            self._docs[doc_id] = FakeDocumentRef()
        return self._docs[doc_id]

    def where(self, field: str, op: str, value: Any) -> FakeQuery:
        """Fake Firestore ``where`` — supports equality only."""
        return FakeQuery(list(self._docs.values()), field, value)


class FakeFirestoreClient:
    """In-memory Firestore fake satisfying :class:`FirestoreClientProtocol`."""

    def __init__(self) -> None:
        self._collections: dict[str, FakeCollectionRef] = {}

    def collection(self, path: str) -> FakeCollectionRef:
        if path not in self._collections:
            self._collections[path] = FakeCollectionRef()
        return self._collections[path]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Default test API key used in tests
TEST_API_KEY = "test-api-key-12345"


@pytest.fixture()
def fake_firestore() -> FakeFirestoreClient:
    """Provide a fresh in-memory Firestore fake."""
    return FakeFirestoreClient()


@pytest.fixture()
def client(fake_firestore: FakeFirestoreClient) -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient with injected fake dependencies.

    Sets up a known API key for authentication tests and resets all
    cached singletons after each test so that no state leaks.
    Uses a generous rate limit so most tests aren't affected by it.
    """
    reset_dependencies()
    reset_api_keys()
    set_firestore_client(fake_firestore)
    # Generous rate limit for normal tests — 10 000 requests / 60 s
    set_rate_limiter(RateLimiter(max_requests=10_000, window_seconds=60))

    # Set known API keys for test isolation
    os.environ["ORNN_API_KEYS"] = TEST_API_KEY
    os.environ.pop("ORNN_REVOKED_API_KEYS", None)

    from api.main import create_app

    app = create_app()
    with TestClient(app) as tc:
        yield tc

    reset_dependencies()
    reset_api_keys()
    # Restore env
    os.environ.pop("ORNN_API_KEYS", None)
    os.environ.pop("ORNN_REVOKED_API_KEYS", None)


@pytest.fixture()
def auth_headers() -> dict[str, str]:
    """Return headers with a valid API key for authenticated requests."""
    return {"X-API-Key": TEST_API_KEY}
