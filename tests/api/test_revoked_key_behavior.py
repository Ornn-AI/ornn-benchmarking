"""Tests for revoked API key rejection across all endpoints.

Covers:
  - VAL-API-009: Revoked key rejection
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
    "report_id": "revoked-test-001",
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

REVOKED_KEY = "revoked-key-xyz-789"


# ---------------------------------------------------------------------------
# VAL-API-009: Revoked key rejection
# ---------------------------------------------------------------------------


class TestRevokedKeyOnIngest:
    """Revoked keys are rejected on POST /api/v1/runs."""

    def test_revoked_key_on_post_returns_401(
        self, client: TestClient
    ) -> None:
        """POST /api/v1/runs with a revoked key returns 401."""
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        resp = client.post(
            "/api/v1/runs",
            json=VALID_PAYLOAD,
            headers={"X-API-Key": REVOKED_KEY},
        )
        assert resp.status_code == 401

        # Clean up
        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()

    def test_revoked_key_post_response_has_no_sensitive_details(
        self, client: TestClient
    ) -> None:
        """401 from revoked key does not expose key details."""
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        resp = client.post(
            "/api/v1/runs",
            json=VALID_PAYLOAD,
            headers={"X-API-Key": REVOKED_KEY},
        )
        body_str = str(resp.json())
        assert REVOKED_KEY not in body_str
        assert TEST_API_KEY not in body_str
        assert "revoked" not in body_str.lower()

        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()

    def test_revoked_key_does_not_persist_data(
        self, client: TestClient, fake_firestore: Any
    ) -> None:
        """When a revoked key is used, nothing is written to Firestore."""
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        client.post(
            "/api/v1/runs",
            json=VALID_PAYLOAD,
            headers={"X-API-Key": REVOKED_KEY},
        )
        collection = fake_firestore.collection("benchmark_runs")
        assert len(collection._docs) == 0

        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()


class TestRevokedKeyOnRetrieval:
    """Revoked keys are rejected on GET /api/v1/runs/{id}."""

    def test_revoked_key_on_get_returns_401(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET /api/v1/runs/{id} with a revoked key returns 401."""
        # First ingest a run with a valid key
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        run_id = resp.json()["run_id"]

        # Now revoke a key and try to retrieve
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": REVOKED_KEY},
        )
        assert resp.status_code == 401

        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()

    def test_revoked_key_get_response_has_no_sensitive_details(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """401 from revoked key on retrieval does not expose key details."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        run_id = resp.json()["run_id"]

        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": REVOKED_KEY},
        )
        body_str = str(resp.json())
        assert REVOKED_KEY not in body_str
        assert TEST_API_KEY not in body_str
        assert "revoked" not in body_str.lower()

        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()

    def test_valid_key_still_works_after_revoked_check(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Valid key works normally even when revoked keys exist."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        run_id = resp.json()["run_id"]

        # Set up revoked keys
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        # Valid key should still work
        resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers=auth_headers,
        )
        assert resp.status_code == 200

        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()


class TestRevokedKeyConsistentBehavior:
    """Revoked key 401 response is indistinguishable from invalid key 401."""

    def test_revoked_and_invalid_key_same_status(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Revoked key and invalid key both return 401 (same status code)."""
        resp = client.post("/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers)
        run_id = resp.json()["run_id"]

        # Test with invalid key
        invalid_resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": "totally-invalid-key"},
        )

        # Set up and test with revoked key
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{REVOKED_KEY}"
        os.environ["ORNN_REVOKED_API_KEYS"] = REVOKED_KEY
        reset_api_keys()

        revoked_resp = client.get(
            f"/api/v1/runs/{run_id}",
            headers={"X-API-Key": REVOKED_KEY},
        )

        assert invalid_resp.status_code == revoked_resp.status_code == 401

        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()
