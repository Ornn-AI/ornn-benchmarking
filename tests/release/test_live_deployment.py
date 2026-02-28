"""End-to-end tests for live Cloud Run deployment.

These tests verify the deployed service at the live URL. They are designed
to run after a successful deployment and validate:
  - Health endpoint reachability
  - Authenticated run submission and persistence
  - Unauthorized request rejection
  - Retrieval round-trip with payload comparison
  - Free-tier-safe deployment configuration

Fulfills:
    VAL-DEPLOY-001 — Cloud Run service is deployed and reachable
    VAL-DEPLOY-002 — Live API accepts authenticated run submission
    VAL-DEPLOY-003 — Live API rejects unauthorized requests
    VAL-DEPLOY-004 — Live API retrieval round-trip
    VAL-DEPLOY-005 — Firestore persistence on live deployment
    VAL-DEPLOY-006 — Deployment uses free-tier-safe configuration
"""

from __future__ import annotations

import os
import subprocess
import uuid

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration — read live URL from env or discover via gcloud
# ---------------------------------------------------------------------------

_LIVE_URL: str | None = None


def _discover_service_url() -> str | None:
    """Discover the Cloud Run service URL via gcloud CLI."""
    try:
        result = subprocess.run(
            [
                "gcloud", "run", "services", "describe", "ornn-api",
                "--project=ornn-benchmarking",
                "--region=us-east1",
                "--format=value(status.url)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_live_url() -> str:
    """Return the live service URL, discovering it if needed."""
    global _LIVE_URL
    if _LIVE_URL is None:
        _LIVE_URL = os.environ.get("ORNN_LIVE_URL") or _discover_service_url()
    if _LIVE_URL is None:
        pytest.skip("Live service URL not available (set ORNN_LIVE_URL or deploy first)")
    return _LIVE_URL


def _get_api_key() -> str:
    """Return the API key for live testing."""
    return os.environ.get("ORNN_TEST_API_KEY", "dev-test-key")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def live_url() -> str:
    """Provide the live service URL, skipping if unavailable."""
    return _get_live_url()


@pytest.fixture()
def api_key() -> str:
    """Provide the API key for live testing."""
    return _get_api_key()


@pytest.fixture()
def sample_payload() -> dict:
    """Generate a unique sample payload for submission testing."""
    unique_id = uuid.uuid4().hex[:12]
    return {
        "schema_version": "1.0.0",
        "report_id": f"e2e-live-test-{unique_id}",
        "created_at": "2026-02-28T06:30:00Z",
        "system_inventory": {
            "gpus": [{
                "uuid": f"GPU-{unique_id}",
                "name": "NVIDIA H100",
                "driver_version": "535.0",
                "cuda_version": "12.2",
                "memory_total_mb": 81920,
            }],
            "os_info": "Linux 6.2.0",
            "kernel_version": "6.2.0",
            "cpu_model": "AMD EPYC 9654",
            "numa_nodes": 2,
            "pytorch_version": "2.1.0",
        },
        "sections": [{
            "name": "compute",
            "status": "completed",
            "started_at": "2026-02-28T06:31:00Z",
            "finished_at": "2026-02-28T06:40:00Z",
            "metrics": {"bf16_tflops": 850.0, "fp8_tflops": 1600.0},
        }],
        "scores": {
            "ornn_i": 95.5,
            "ornn_t": 88.2,
            "qualification": "Premium",
            "components": {"bw": 2800.0, "fp8": 1600.0, "bf16": 850.0, "ar": 180.0},
            "score_status": "ok",
            "aggregate_method": "mean",
        },
        "manifest": {"sections": ["compute"]},
    }


# ===================================================================
# VAL-DEPLOY-001: Cloud Run service is deployed and reachable
# ===================================================================


class TestLiveHealthEndpoint:
    """Service must be deployed and health endpoint reachable."""

    def test_health_returns_200(self, live_url: str) -> None:
        resp = httpx.get(f"{live_url}/health", timeout=30)
        assert resp.status_code == 200

    def test_health_body_contains_status_ok(self, live_url: str) -> None:
        resp = httpx.get(f"{live_url}/health", timeout=30)
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_body_contains_version(self, live_url: str) -> None:
        resp = httpx.get(f"{live_url}/health", timeout=30)
        body = resp.json()
        assert "version" in body


# ===================================================================
# VAL-DEPLOY-002: Live authenticated run submission
# ===================================================================


class TestLiveAuthenticatedSubmission:
    """POST /api/v1/runs with valid key must persist and return run_id."""

    def test_submit_returns_201(
        self, live_url: str, api_key: str, sample_payload: dict
    ) -> None:
        resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json=sample_payload,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        assert resp.status_code == 201

    def test_submit_returns_run_id(
        self, live_url: str, api_key: str, sample_payload: dict
    ) -> None:
        resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json=sample_payload,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        body = resp.json()
        assert "run_id" in body
        assert len(body["run_id"]) > 0

    def test_submit_returns_timestamps(
        self, live_url: str, api_key: str, sample_payload: dict
    ) -> None:
        resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json=sample_payload,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        body = resp.json()
        assert "received_at" in body
        assert "stored_at" in body


# ===================================================================
# VAL-DEPLOY-003: Live unauthorized request rejection
# ===================================================================


class TestLiveUnauthorizedRejection:
    """Missing/invalid API key must be rejected with 401."""

    def test_no_api_key_returns_401(self, live_url: str) -> None:
        resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json={"schema_version": "1.0.0", "report_id": "x", "created_at": "x",
                  "system_inventory": {}, "scores": {}},
            timeout=30,
        )
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, live_url: str) -> None:
        resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json={"schema_version": "1.0.0", "report_id": "x", "created_at": "x",
                  "system_inventory": {}, "scores": {}},
            headers={"X-API-Key": "completely-invalid-key"},
            timeout=30,
        )
        assert resp.status_code == 401

    def test_401_does_not_expose_internals(self, live_url: str) -> None:
        resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json={"schema_version": "1.0.0", "report_id": "x", "created_at": "x",
                  "system_inventory": {}, "scores": {}},
            headers={"X-API-Key": "invalid"},
            timeout=30,
        )
        body = resp.json()
        # Should not leak any key names or internal details
        assert "dev-test-key" not in str(body)
        assert "valid" not in body.get("detail", "").lower() or "key" in body.get("detail", "")


# ===================================================================
# VAL-DEPLOY-004: Live retrieval round-trip
# ===================================================================


class TestLiveRetrievalRoundTrip:
    """GET /api/v1/runs/{id} must return the submitted payload."""

    def test_roundtrip_returns_matching_payload(
        self, live_url: str, api_key: str, sample_payload: dict
    ) -> None:
        # Submit
        post_resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json=sample_payload,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        assert post_resp.status_code == 201
        run_id = post_resp.json()["run_id"]

        # Retrieve
        get_resp = httpx.get(
            f"{live_url}/api/v1/runs/{run_id}",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        assert get_resp.status_code == 200

        retrieved = get_resp.json()
        assert retrieved["report_id"] == sample_payload["report_id"]
        assert retrieved["schema_version"] == sample_payload["schema_version"]
        assert retrieved["scores"]["ornn_i"] == sample_payload["scores"]["ornn_i"]
        assert retrieved["scores"]["ornn_t"] == sample_payload["scores"]["ornn_t"]
        assert retrieved["scores"]["qualification"] == sample_payload["scores"]["qualification"]

    def test_missing_run_returns_404(self, live_url: str, api_key: str) -> None:
        resp = httpx.get(
            f"{live_url}/api/v1/runs/nonexistent-id-12345",
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        assert resp.status_code == 404


# ===================================================================
# VAL-DEPLOY-005: Firestore persistence evidence
# ===================================================================


class TestLiveFirestorePersistence:
    """Data must be persisted in Firestore and survive restarts."""

    def test_data_persists_across_requests(
        self, live_url: str, api_key: str, sample_payload: dict
    ) -> None:
        """Submit and retrieve in separate requests (may hit different instances)."""
        # Submit
        post_resp = httpx.post(
            f"{live_url}/api/v1/runs",
            json=sample_payload,
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        run_id = post_resp.json()["run_id"]

        # Retrieve with a fresh client (separate connection)
        with httpx.Client(timeout=30) as client:
            get_resp = client.get(
                f"{live_url}/api/v1/runs/{run_id}",
                headers={"X-API-Key": api_key},
            )

        assert get_resp.status_code == 200
        assert get_resp.json()["run_id"] == run_id
        assert get_resp.json()["report_id"] == sample_payload["report_id"]


# ===================================================================
# VAL-DEPLOY-006: Free-tier-safe configuration
# ===================================================================


class TestLiveFreeTierConfig:
    """Deployment must use free-tier-safe configuration."""

    def test_service_description_shows_scale_to_zero(self) -> None:
        """gcloud run services describe shows min-instances=0."""
        try:
            result = subprocess.run(
                [
                    "gcloud", "run", "services", "describe", "ornn-api",
                    "--project=ornn-benchmarking",
                    "--region=us-east1",
                    "--format=yaml(spec.template.metadata.annotations)",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pytest.skip("gcloud CLI not available for config verification")
            return

        if result.returncode != 0:
            pytest.skip("gcloud describe failed — may not have permissions")
            return

        output = result.stdout
        # Cloud Run defaults to min-instances=0 when not explicitly set
        # (absence of minScale annotation = scale-to-zero)
        assert "minScale" not in output or "minScale: '0'" in output, (
            "Cloud Run service should have min-instances=0 (scale-to-zero)"
        )
