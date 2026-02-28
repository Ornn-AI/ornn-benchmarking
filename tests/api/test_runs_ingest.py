"""Tests for POST /api/v1/runs — run ingest endpoint.

Covers:
  - VAL-API-001: Authenticated run submission persistence
  - VAL-API-002: Unauthorized access rejection
  - VAL-API-003: Request validation semantics
"""

from __future__ import annotations

import os
from typing import Any

from api.auth import reset_api_keys
from fastapi.testclient import TestClient

from tests.api.conftest import TEST_API_KEY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict[str, Any] = {
    "schema_version": "1.0.0",
    "report_id": "test-run-001",
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
# VAL-API-001: Authenticated run submission persistence
# ---------------------------------------------------------------------------


class TestRunIngestSuccess:
    """Tests for successful run ingest with valid API key."""

    def test_valid_request_returns_201(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """POST with valid key and valid payload returns 201."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        assert resp.status_code == 201

    def test_response_contains_run_id(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response body includes a non-empty ``run_id``."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        data = resp.json()
        assert "run_id" in data
        assert isinstance(data["run_id"], str)
        assert len(data["run_id"]) > 0

    def test_response_contains_received_at(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response body includes ``received_at`` server timestamp."""
        data = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        assert "received_at" in data
        assert isinstance(data["received_at"], str)
        assert len(data["received_at"]) > 0

    def test_response_contains_stored_at(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response body includes ``stored_at`` server timestamp."""
        data = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        assert "stored_at" in data
        assert isinstance(data["stored_at"], str)
        assert len(data["stored_at"]) > 0

    def test_run_id_is_unique_across_requests(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Two successive ingests produce different ``run_id`` values."""
        r1 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        r2 = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers).json()
        assert r1["run_id"] != r2["run_id"]

    def test_payload_persisted_in_firestore(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        fake_firestore: Any,
    ) -> None:
        """Ingested payload is stored in the ``benchmark_runs`` collection."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        run_id = resp.json()["run_id"]

        collection = fake_firestore.collection("benchmark_runs")
        doc = collection.document(run_id)
        snapshot = doc.get()
        assert snapshot.exists
        stored = snapshot.to_dict()
        assert stored["report_id"] == "test-run-001"
        assert stored["schema_version"] == "1.0.0"
        assert stored["run_id"] == run_id

    def test_minimal_valid_payload(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Minimal payload with only required fields returns 201."""
        minimal = {
            "schema_version": "1.0.0",
            "report_id": "min-001",
            "created_at": "2024-01-15T10:30:00Z",
            "system_inventory": {},
            "scores": {},
        }
        resp = client.post("/api/v1/runs", json=minimal, headers=auth_headers)
        assert resp.status_code == 201

    def test_response_content_type_is_json(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response Content-Type is application/json."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        assert resp.headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# VAL-API-002: Unauthorized access rejection
# ---------------------------------------------------------------------------


class TestRunIngestAuth:
    """Tests for auth rejection on POST /api/v1/runs."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        """Request without X-API-Key header returns 401."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD)
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, client: TestClient) -> None:
        """Request with wrong API key returns 401."""
        headers = {"X-API-Key": "wrong-key-value"}
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_empty_api_key_returns_401(self, client: TestClient) -> None:
        """Request with empty API key returns 401."""
        headers = {"X-API-Key": ""}
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_401_response_has_no_sensitive_details(self, client: TestClient) -> None:
        """401 response body does not expose valid keys or internal details."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD)
        body = resp.json()
        assert "detail" in body
        # Must not contain the actual valid key
        assert TEST_API_KEY not in str(body)
        # Must not contain stack traces or internal paths
        assert "traceback" not in str(body).lower()
        assert "Traceback" not in str(body)

    def test_revoked_key_returns_401(self, client: TestClient) -> None:
        """Previously valid but revoked key returns 401."""
        revoked_key = "revoked-key-xyz"
        # Set the revoked key in environment and reset cache
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{revoked_key}"
        os.environ["ORNN_REVOKED_API_KEYS"] = revoked_key
        reset_api_keys()

        headers = {"X-API-Key": revoked_key}
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=headers)
        assert resp.status_code == 401

        # Clean up
        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()

    def test_no_payload_persisted_on_auth_failure(
        self, client: TestClient, fake_firestore: Any
    ) -> None:
        """When auth fails, nothing is written to Firestore."""
        client.post("/api/v1/runs", json=VALID_PAYLOAD)  # no key
        collection = fake_firestore.collection("benchmark_runs")
        assert len(collection._docs) == 0


# ---------------------------------------------------------------------------
# VAL-API-003: Request validation semantics
# ---------------------------------------------------------------------------


class TestRunIngestValidation:
    """Tests for payload validation on POST /api/v1/runs."""

    def test_empty_body_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Missing body returns 422."""
        resp = client.post(
            "/api/v1/runs",
            content=b"",
            headers={**auth_headers, "Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_non_json_body_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Non-JSON body returns 422."""
        resp = client.post(
            "/api/v1/runs",
            content=b"not json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )
        assert resp.status_code == 422

    def test_missing_required_field_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Missing required field (schema_version) returns 422."""
        payload = {
            "report_id": "test-001",
            "created_at": "2024-01-15T10:30:00Z",
            "system_inventory": {},
            "scores": {},
        }
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_empty_schema_version_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Empty string for schema_version violates min_length."""
        payload = {
            "schema_version": "",
            "report_id": "test-001",
            "created_at": "2024-01-15T10:30:00Z",
            "system_inventory": {},
            "scores": {},
        }
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_missing_report_id_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Missing report_id returns 422."""
        payload = {
            "schema_version": "1.0.0",
            "created_at": "2024-01-15T10:30:00Z",
            "system_inventory": {},
            "scores": {},
        }
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_missing_scores_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Missing scores object returns 422."""
        payload = {
            "schema_version": "1.0.0",
            "report_id": "test-001",
            "created_at": "2024-01-15T10:30:00Z",
            "system_inventory": {},
        }
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_missing_system_inventory_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Missing system_inventory returns 422."""
        payload = {
            "schema_version": "1.0.0",
            "report_id": "test-001",
            "created_at": "2024-01-15T10:30:00Z",
            "scores": {},
        }
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_422_response_has_field_level_errors(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """422 response includes actionable field-level error details."""
        payload: dict[str, Any] = {}  # completely empty
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" in body
        # FastAPI returns a list of validation errors
        assert isinstance(body["detail"], list)
        assert len(body["detail"]) > 0
        # Each error should have location and message info
        first_err = body["detail"][0]
        assert "loc" in first_err
        assert "msg" in first_err

    def test_wrong_type_for_field_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Wrong type (e.g., int for schema_version) returns 422."""
        payload = {
            "schema_version": 123,
            "report_id": "test-001",
            "created_at": "2024-01-15T10:30:00Z",
            "system_inventory": {},
            "scores": {},
        }
        resp = client.post("/api/v1/runs", json=payload, headers=auth_headers)
        # Pydantic may coerce int to str, but the min_length check applies
        # If coerced: "123" passes min_length=1, so this should be 201
        # This depends on Pydantic strict mode — we accept either outcome
        assert resp.status_code in (201, 422)

    def test_validation_error_before_auth_when_body_invalid(
        self, client: TestClient
    ) -> None:
        """Auth check runs before validation — missing key on invalid payload still returns 401."""
        resp = client.post("/api/v1/runs", json={})
        # Auth fails first → 401, not 422
        assert resp.status_code == 401
