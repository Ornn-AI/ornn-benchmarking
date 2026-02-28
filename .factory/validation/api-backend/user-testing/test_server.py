"""Lightweight test API server with in-memory Firestore fake.

Starts the FastAPI app on port 8080 with injected fake dependencies
so that user testing can be performed via curl without requiring
a real Firestore emulator or Java runtime.

Usage:
  cd <project-root>
  .venv/bin/python -m factory_test_server
  # or:
  PYTHONPATH=. .venv/bin/python .factory/validation/api-backend/user-testing/test_server.py
"""
from __future__ import annotations

import os
import sys
import uuid
from collections.abc import Generator
from typing import Any

# Ensure project root is on sys.path so `api` and `tests` packages resolve
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "..", "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Configure test API keys BEFORE importing API modules
os.environ["ORNN_API_KEYS"] = "test-key-1,test-key-2,test-key-3,test-key-4"
os.environ["ORNN_REVOKED_API_KEYS"] = "revoked-key-1"
# Rate limit: 3 requests per 10 seconds per key — low enough to test 429,
# but short enough window that other groups can wait it out.
os.environ["RATE_LIMIT_REQUESTS"] = "3"
os.environ["RATE_LIMIT_WINDOW_SECONDS"] = "10"

from api.auth import reset_api_keys  # noqa: E402
from api.dependencies import (  # noqa: E402
    reset_dependencies,
    set_firestore_client,
    set_rate_limiter,
)
from api.rate_limit import RateLimiter  # noqa: E402

# ---------------------------------------------------------------------------
# Inline Firestore fake (copied from tests/api/conftest.py to avoid test dep)
# ---------------------------------------------------------------------------


class FakeDocumentRef:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def set(self, data: dict[str, Any]) -> None:
        self._data = data

    def get(self) -> "FakeDocumentRef":
        return self

    @property
    def exists(self) -> bool:
        return bool(self._data)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeQueryResult:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


class FakeQuery:
    def __init__(self, docs: list[FakeDocumentRef], field: str, value: Any) -> None:
        self._docs = docs
        self._field = field
        self._value = value
        self._limit: int | None = None

    def limit(self, n: int) -> "FakeQuery":
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
    def __init__(self) -> None:
        self._docs: dict[str, FakeDocumentRef] = {}

    def document(self, doc_id: str | None = None) -> FakeDocumentRef:
        if doc_id is None:
            doc_id = uuid.uuid4().hex
        if doc_id not in self._docs:
            self._docs[doc_id] = FakeDocumentRef()
        return self._docs[doc_id]

    def where(self, field: str, op: str, value: Any) -> FakeQuery:
        return FakeQuery(list(self._docs.values()), field, value)


class FakeFirestoreClient:
    def __init__(self) -> None:
        self._collections: dict[str, FakeCollectionRef] = {}

    def collection(self, path: str) -> FakeCollectionRef:
        if path not in self._collections:
            self._collections[path] = FakeCollectionRef()
        return self._collections[path]


# ---------------------------------------------------------------------------
# Wire up dependencies
# ---------------------------------------------------------------------------

reset_dependencies()
reset_api_keys()

fake_db = FakeFirestoreClient()
set_firestore_client(fake_db)
set_rate_limiter(RateLimiter(max_requests=3, window_seconds=10))

from api.main import create_app  # noqa: E402
import uvicorn  # noqa: E402

app = create_app()

if __name__ == "__main__":
    print("Starting test API server on port 8080 with fake Firestore...")
    print("Valid API keys: test-key-1, test-key-2, test-key-3")
    print("Revoked API key: revoked-key-1")
    print("Rate limit: 5 requests per 60 seconds per key")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
