"""Tests for idempotent ingest (dedupe) on POST /api/v1/runs.

Covers:
  - VAL-API-006: Idempotent ingest contract
  - VAL-CROSS-003: Upload retry idempotency flow
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict[str, Any] = {
    "schema_version": "1.0.0",
    "report_id": "dedupe-test-001",
    "created_at": "2024-01-15T10:30:00Z",
    "system_inventory": {
        "gpus": [
            {
                "uuid": "GPU-12345678-abcd-1234-abcd-123456789abc",
                "name": "NVIDIA H100 80GB HBM3",
                "driver_version": "535.129.03",
                "cuda_version": "12.2",
                "memory_total_mb": 81920,
            }
        ],
        "os_info": "Ubuntu 22.04.3 LTS",
        "kernel_version": "5.15.0-91-generic",
        "cpu_model": "Intel(R) Xeon(R) Platinum 8480+",
        "numa_nodes": 2,
        "pytorch_version": "2.1.2",
    },
    "sections": [
        {
            "name": "compute",
            "status": "completed",
            "started_at": "2024-01-15T10:30:01Z",
            "finished_at": "2024-01-15T10:35:00Z",
        }
    ],
    "scores": {
        "ornn_i": 85.5,
        "ornn_t": 78.2,
        "qualification": "Premium",
        "components": {"bw": 1.2, "fp8": 0.9, "bf16": 1.1, "ar": 0.8},
        "score_status": "valid",
    },
}


# ---------------------------------------------------------------------------
# VAL-API-006: Idempotent ingest contract
# ---------------------------------------------------------------------------


class TestIdempotentIngestSamePayload:
    """Duplicate uploads of identical payloads must not create duplicates."""

    def test_first_upload_returns_201(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """First POST with a new payload returns 201 Created."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        assert resp.status_code == 201

    def test_second_upload_returns_200(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Second POST with identical payload returns 200 (not 201)."""
        client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        assert resp.status_code == 200

    def test_duplicate_returns_same_run_id(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Duplicate upload returns the same ``run_id`` as the first."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        r2 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        assert r1["run_id"] == r2["run_id"]

    def test_duplicate_returns_original_timestamps(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Duplicate upload returns original ``received_at`` and ``stored_at``."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        r2 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        assert r1["received_at"] == r2["received_at"]
        assert r1["stored_at"] == r2["stored_at"]

    def test_triple_upload_still_returns_same_run_id(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Three identical uploads all reference the same ``run_id``."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        r2 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        r3 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        assert r1["run_id"] == r2["run_id"] == r3["run_id"]

    def test_no_duplicate_document_in_firestore(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        fake_firestore: Any,
    ) -> None:
        """Only one document is stored despite multiple identical uploads."""
        client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)

        collection = fake_firestore.collection("benchmark_runs")
        stored_docs = [
            doc for doc in collection._docs.values() if doc.exists
        ]
        assert len(stored_docs) == 1

    def test_retrieval_after_duplicate_returns_original(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET after duplicate upload returns the original payload data."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)

        resp = client.get(f"/api/v1/runs/{r1['run_id']}", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["report_id"] == VALID_PAYLOAD["report_id"]


class TestIdempotentIngestDifferentPayloads:
    """Distinct payloads must produce distinct runs."""

    def test_different_report_id_creates_new_run(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Different ``report_id`` creates a separate run."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()

        different = {**VALID_PAYLOAD, "report_id": "dedupe-test-002"}
        r2 = client.post("/api/v1/runs", json=different, headers=auth_headers).json()
        assert r1["run_id"] != r2["run_id"]

    def test_different_created_at_creates_new_run(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Different ``created_at`` creates a separate run."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()

        different = {**VALID_PAYLOAD, "created_at": "2024-01-16T12:00:00Z"}
        r2 = client.post("/api/v1/runs", json=different, headers=auth_headers).json()
        assert r1["run_id"] != r2["run_id"]

    def test_unsupported_schema_version_is_rejected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Unsupported ``schema_version`` is rejected with 422 (VAL-CROSS-005)."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        assert r1.status_code == 201

        different = {**VALID_PAYLOAD, "schema_version": "2.0.0"}
        r2 = client.post("/api/v1/runs", json=different, headers=auth_headers)
        assert r2.status_code == 422
        detail = r2.json().get("detail", "")
        assert "schema version" in detail.lower() or "2.0.0" in detail

    def test_both_distinct_runs_stored_in_firestore(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        fake_firestore: Any,
    ) -> None:
        """Two distinct payloads each get their own Firestore document."""
        client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        different = {**VALID_PAYLOAD, "report_id": "dedupe-test-different"}
        client.post("/api/v1/runs", json=different, headers=auth_headers)

        collection = fake_firestore.collection("benchmark_runs")
        stored_docs = [
            doc for doc in collection._docs.values() if doc.exists
        ]
        assert len(stored_docs) == 2


class TestDedupeKeyDeterminism:
    """The dedupe key is deterministic based on identity fields only."""

    def test_same_identity_different_scores_dedupe(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Same identity fields but different scores still dedupes (retry scenario)."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()

        # Same identity fields, different score values (simulates retry)
        modified = dict(VALID_PAYLOAD)
        modified["scores"] = {
            "ornn_i": 90.0,
            "ornn_t": 80.0,
            "qualification": "Premium",
            "components": {"bw": 1.5, "fp8": 1.0},
            "score_status": "valid",
        }
        r2 = client.post("/api/v1/runs", json=modified, headers=auth_headers).json()

        # Same identity → same run_id (dedupe hit)
        assert r1["run_id"] == r2["run_id"]

    def test_duplicate_response_body_structure(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Deduplicated response has the same structure as a fresh ingest."""
        client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        data = resp.json()
        assert "run_id" in data
        assert "received_at" in data
        assert "stored_at" in data
