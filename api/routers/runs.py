"""Router for benchmark run ingest and retrieval.

Endpoints:
  - ``POST /api/v1/runs``     — ingest a benchmark run (idempotent)
  - ``GET  /api/v1/runs/{id}`` — retrieve a stored run by id
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from api.auth import require_api_key
from api.dependencies import (
    FirestoreClientProtocol,
    get_firestore_client,
    get_rate_limiter,
)
from api.models import RunPayload, RunResponse
from api.rate_limit import RateLimiter

router = APIRouter(prefix="/api/v1", tags=["runs"])

COLLECTION_NAME = "benchmark_runs"

# Supported schema versions — unsupported versions are rejected with 422
# and explicit upgrade/downgrade guidance (VAL-CROSS-005).
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}

_require_api_key = Depends(require_api_key)
_get_firestore_client = Depends(get_firestore_client)
_get_rate_limiter = Depends(get_rate_limiter)


def _compute_dedupe_key(payload: RunPayload) -> str:
    """Derive a deterministic dedupe key from the payload identity fields.

    Uses ``report_id``, ``created_at``, and ``schema_version`` to form a
    stable hash.  Identical logical runs (same report uploaded twice) will
    always produce the same key.
    """
    identity = json.dumps(
        {
            "report_id": payload.report_id,
            "created_at": payload.created_at,
            "schema_version": payload.schema_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


def _find_existing_by_dedupe_key(
    db: FirestoreClientProtocol,
    dedupe_key: str,
) -> dict[str, Any] | None:
    """Look up an existing document by its ``dedupe_key`` field.

    Returns the document dict if found, ``None`` otherwise.
    """
    collection: Any = db.collection(COLLECTION_NAME)
    query = collection.where("dedupe_key", "==", dedupe_key).limit(1)
    docs = list(query.stream())
    if docs:
        result: dict[str, Any] = docs[0].to_dict()
        return result
    return None


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_run(
    payload: RunPayload,
    request: Request,
    response: Response,
    api_key: str = _require_api_key,
    db: FirestoreClientProtocol = _get_firestore_client,
    limiter: RateLimiter = _get_rate_limiter,
) -> RunResponse:
    """Ingest a benchmark run (idempotent).

    Validates the request payload via Pydantic, authenticates via API key,
    checks rate limits, deduplicates against existing runs, persists to
    Firestore, and returns a server-generated ``run_id`` with timestamps.

    If an identical run (same ``report_id`` + ``created_at`` +
    ``schema_version``) has already been ingested, returns the existing
    ``run_id`` with HTTP 200 instead of creating a duplicate.
    """
    # --- Schema version check (VAL-CROSS-005) --------------------------
    if payload.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unsupported schema version '{payload.schema_version}'. "
                f"Supported versions: {supported}. "
                f"Please upgrade or downgrade ornn-bench to a compatible version."
            ),
        )

    # --- Rate limiting --------------------------------------------------
    allowed, retry_after = limiter.check(api_key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    # --- Idempotent dedupe ----------------------------------------------
    dedupe_key = _compute_dedupe_key(payload)
    existing = _find_existing_by_dedupe_key(db, dedupe_key)
    if existing is not None:
        # Return existing run data with 200 instead of 201
        response.status_code = status.HTTP_200_OK
        return RunResponse(
            run_id=existing["run_id"],
            received_at=existing["received_at"],
            stored_at=existing["stored_at"],
        )

    # --- Persist new run ------------------------------------------------
    run_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    doc_data: dict[str, Any] = {
        "run_id": run_id,
        "received_at": now,
        "stored_at": now,
        "dedupe_key": dedupe_key,
        **payload.model_dump(),
    }

    collection = db.collection(COLLECTION_NAME)
    doc_ref = collection.document(run_id)
    doc_ref.set(doc_data)

    return RunResponse(
        run_id=run_id,
        received_at=now,
        stored_at=now,
    )


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    api_key: str = _require_api_key,
    db: FirestoreClientProtocol = _get_firestore_client,
    limiter: RateLimiter = _get_rate_limiter,
) -> dict[str, Any]:
    """Retrieve a stored benchmark run by its id.

    Requires a valid API key.  Returns 404 if the run does not exist.
    The response body is the full stored document (payload + server fields).
    """
    # --- Rate limiting --------------------------------------------------
    allowed, retry_after = limiter.check(api_key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    collection = db.collection(COLLECTION_NAME)
    doc_ref = collection.document(run_id)
    snapshot = doc_ref.get()

    if not snapshot.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found.",
        )

    data: dict[str, Any] = snapshot.to_dict()
    return data
