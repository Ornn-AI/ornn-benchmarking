"""Router for benchmark run ingest (``POST /api/v1/runs``)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, status

from api.auth import require_api_key
from api.dependencies import FirestoreClientProtocol, get_firestore_client
from api.models import RunPayload, RunResponse

router = APIRouter(prefix="/api/v1", tags=["runs"])

COLLECTION_NAME = "benchmark_runs"

_require_api_key = Depends(require_api_key)
_get_firestore_client = Depends(get_firestore_client)


@router.post(
    "/runs",
    response_model=RunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_run(
    payload: RunPayload,
    api_key: str = _require_api_key,
    db: FirestoreClientProtocol = _get_firestore_client,
) -> RunResponse:
    """Ingest a benchmark run.

    Validates the request payload via Pydantic, authenticates via API key,
    persists to Firestore, and returns a server-generated ``run_id`` with
    timestamps.
    """
    run_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc).isoformat()

    doc_data = {
        "run_id": run_id,
        "received_at": now,
        "stored_at": now,
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
