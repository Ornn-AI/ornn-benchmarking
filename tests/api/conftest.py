"""Shared fixtures for API tests."""

from __future__ import annotations

import uuid
from collections.abc import Generator
from typing import Any

import pytest
from api.dependencies import reset_dependencies, set_firestore_client
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


@pytest.fixture()
def fake_firestore() -> FakeFirestoreClient:
    """Provide a fresh in-memory Firestore fake."""
    return FakeFirestoreClient()


@pytest.fixture()
def client(fake_firestore: FakeFirestoreClient) -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient with injected fake dependencies.

    Dependencies are reset after each test so that no state leaks.
    """
    reset_dependencies()
    set_firestore_client(fake_firestore)

    from api.main import create_app

    app = create_app()
    with TestClient(app) as tc:
        yield tc

    reset_dependencies()
