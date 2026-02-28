"""Tests for GET /api/v1/runs/{id} — run retrieval endpoint.

Covers:
  - VAL-API-004: Retrieval semantics and access control
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.api.conftest import TEST_API_KEY

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict[str, Any] = {
    "schema_version": "1.0.0",
    "report_id": "retrieval-test-001",
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


def _ingest_run(
    client: TestClient, auth_headers: dict[str, str]
) -> str:
    """Helper: ingest a run and return its ``run_id``."""
    resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
    assert resp.status_code == 201, f"Ingest failed: {resp.text}"
    return resp.json()["run_id"]


# ---------------------------------------------------------------------------
# VAL-API-004: Retrieval semantics and access control
# ---------------------------------------------------------------------------


class TestRunRetrievalSuccess:
    """Tests for successful run retrieval with valid API key."""

    def test_get_existing_run_returns_200(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET /api/v1/runs/{id} for an existing run returns 200."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)
        assert resp.status_code == 200

    def test_response_contains_stored_payload_fields(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response body contains the original payload fields."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)
        data = resp.json()
        assert data["run_id"] == run_id
        assert data["schema_version"] == VALID_PAYLOAD["schema_version"]
        assert data["report_id"] == VALID_PAYLOAD["report_id"]
        assert data["created_at"] == VALID_PAYLOAD["created_at"]

    def test_response_contains_server_timestamps(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response includes server-assigned ``received_at`` and ``stored_at``."""
        run_id = _ingest_run(client, auth_headers)
        data = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers).json()
        assert "received_at" in data
        assert "stored_at" in data
        assert isinstance(data["received_at"], str)
        assert isinstance(data["stored_at"], str)

    def test_response_contains_scores(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response body contains the scores object from the original payload."""
        run_id = _ingest_run(client, auth_headers)
        data = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers).json()
        assert "scores" in data
        scores = data["scores"]
        assert scores["ornn_i"] == 85.5
        assert scores["ornn_t"] == 78.2
        assert scores["qualification"] == "Premium"

    def test_response_contains_system_inventory(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response body contains system_inventory from the original payload."""
        run_id = _ingest_run(client, auth_headers)
        data = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers).json()
        assert "system_inventory" in data
        inv = data["system_inventory"]
        assert len(inv["gpus"]) == 1
        assert inv["gpus"][0]["name"] == "NVIDIA H100 80GB HBM3"

    def test_response_content_type_is_json(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response Content-Type is application/json."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)
        assert resp.headers["content-type"] == "application/json"


class TestRunRetrievalNotFound:
    """Tests for 404 on missing run id."""

    def test_nonexistent_id_returns_404(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET /api/v1/runs/{id} with a nonexistent id returns 404."""
        resp = client.get("/api/v1/runs/nonexistent-id-12345", headers=auth_headers)
        assert resp.status_code == 404

    def test_404_response_has_detail(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """404 response contains a descriptive ``detail`` field."""
        resp = client.get("/api/v1/runs/nonexistent-id-12345", headers=auth_headers)
        body = resp.json()
        assert "detail" in body
        assert isinstance(body["detail"], str)
        assert len(body["detail"]) > 0

    def test_404_does_not_expose_internals(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """404 response does not expose internal details or stack traces."""
        resp = client.get("/api/v1/runs/nonexistent-id-12345", headers=auth_headers)
        body_str = str(resp.json())
        assert "traceback" not in body_str.lower()
        assert "Traceback" not in body_str


class TestRunRetrievalAuth:
    """Tests for auth rejection on GET /api/v1/runs/{id}."""

    def test_missing_api_key_returns_401(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET without X-API-Key returns 401."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(f"/api/v1/runs/{run_id}")
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET with invalid API key returns 401."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": "wrong-key-value"},
        )
        assert resp.status_code == 401

    def test_empty_api_key_returns_401(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET with empty API key returns 401."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": ""},
        )
        assert resp.status_code == 401

    def test_401_response_has_no_sensitive_details(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """401 response body does not expose valid keys or internals."""
        run_id = _ingest_run(client, auth_headers)
        resp = client.get(f"/api/v1/runs/{run_id}")
        body = resp.json()
        assert "detail" in body
        assert TEST_API_KEY not in str(body)
        assert "traceback" not in str(body).lower()

    def test_401_before_404_on_missing_id(
        self, client: TestClient
    ) -> None:
        """Auth rejection happens before 404 — missing key on missing id returns 401."""
        resp = client.get("/api/v1/runs/nonexistent-id")
        assert resp.status_code == 401
