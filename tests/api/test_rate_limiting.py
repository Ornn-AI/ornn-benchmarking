"""Tests for per-key API rate limiting with 429 semantics.

Covers:
  - VAL-API-010: Rate limiting behavior
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

import pytest
from api.auth import reset_api_keys
from api.dependencies import reset_dependencies, set_firestore_client, set_rate_limiter
from api.rate_limit import RateLimiter
from fastapi.testclient import TestClient

from tests.api.conftest import TEST_API_KEY, FakeFirestoreClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_PAYLOAD: dict[str, Any] = {
    "schema_version": "1.0.0",
    "report_id": "rate-limit-test-001",
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

SECOND_API_KEY = "second-test-key-67890"


@pytest.fixture()
def rate_limited_client() -> Generator[TestClient, None, None]:
    """TestClient with a tight rate limit (3 requests / 60 s) for testing."""
    reset_dependencies()
    reset_api_keys()
    fake_fs = FakeFirestoreClient()
    set_firestore_client(fake_fs)
    set_rate_limiter(RateLimiter(max_requests=3, window_seconds=60))

    os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{SECOND_API_KEY}"
    os.environ.pop("ORNN_REVOKED_API_KEYS", None)
    reset_api_keys()

    from api.main import create_app

    app = create_app()
    with TestClient(app) as tc:
        yield tc

    reset_dependencies()
    reset_api_keys()
    os.environ.pop("ORNN_API_KEYS", None)
    os.environ.pop("ORNN_REVOKED_API_KEYS", None)


# ---------------------------------------------------------------------------
# VAL-API-010: Rate limiting behavior
# ---------------------------------------------------------------------------


class TestRateLimitOnPost:
    """Rate limiting on POST /api/v1/runs."""

    def test_requests_within_limit_succeed(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Requests within the limit return success (201 or 200)."""
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-post-{i}"}
            resp = rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=auth_headers
            )
            assert resp.status_code in (200, 201), f"Request {i} failed: {resp.status_code}"

    def test_exceeding_limit_returns_429(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Request beyond the limit returns 429."""
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-post-over-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=auth_headers
            )

        payload = {**VALID_PAYLOAD, "report_id": "rl-post-over-3"}
        resp = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=auth_headers
        )
        assert resp.status_code == 429

    def test_429_response_has_retry_after_header(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """429 response includes a ``Retry-After`` header."""
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-post-hdr-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=auth_headers
            )

        payload = {**VALID_PAYLOAD, "report_id": "rl-post-hdr-3"}
        resp = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=auth_headers
        )
        assert resp.status_code == 429
        assert "retry-after" in resp.headers
        retry_after = int(resp.headers["retry-after"])
        assert retry_after > 0

    def test_429_response_body_has_detail(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """429 response body contains a descriptive message."""
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-post-body-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=auth_headers
            )

        payload = {**VALID_PAYLOAD, "report_id": "rl-post-body-3"}
        resp = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=auth_headers
        )
        body = resp.json()
        assert "detail" in body
        assert "rate limit" in body["detail"].lower() or "retry" in body["detail"].lower()


class TestRateLimitOnGet:
    """Rate limiting on GET /api/v1/runs/{id}."""

    def test_get_exceeding_limit_returns_429(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET requests beyond the limit return 429."""
        # First ingest a run (uses 1 of 3 slots)
        resp = rate_limited_client.post(
            "/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers
        )
        run_id = resp.json()["run_id"]

        # Use remaining 2 slots on GET
        rate_limited_client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)
        rate_limited_client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)

        # 4th request should be rate-limited
        resp = rate_limited_client.get(
            f"/api/v1/runs/{run_id}", headers=auth_headers
        )
        assert resp.status_code == 429

    def test_get_429_has_retry_after(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """GET 429 includes Retry-After header."""
        resp = rate_limited_client.post(
            "/api/v1/runs", json=VALID_PAYLOAD, headers=auth_headers
        )
        run_id = resp.json()["run_id"]

        rate_limited_client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)
        rate_limited_client.get(f"/api/v1/runs/{run_id}", headers=auth_headers)

        resp = rate_limited_client.get(
            f"/api/v1/runs/{run_id}", headers=auth_headers
        )
        assert resp.status_code == 429
        assert "retry-after" in resp.headers


class TestRateLimitPerKey:
    """Rate limits are enforced per API key."""

    def test_different_keys_have_separate_limits(
        self, rate_limited_client: TestClient
    ) -> None:
        """Key A hitting limit does not affect Key B."""
        headers_a = {"X-API-Key": TEST_API_KEY}
        headers_b = {"X-API-Key": SECOND_API_KEY}

        # Exhaust key A's limit
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-keya-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=headers_a
            )

        # Key A should be rate-limited
        payload = {**VALID_PAYLOAD, "report_id": "rl-keya-3"}
        resp_a = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=headers_a
        )
        assert resp_a.status_code == 429

        # Key B should still work fine
        payload = {**VALID_PAYLOAD, "report_id": "rl-keyb-0"}
        resp_b = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=headers_b
        )
        assert resp_b.status_code in (200, 201)

    def test_key_b_not_affected_after_key_a_exhausted(
        self, rate_limited_client: TestClient
    ) -> None:
        """After key A is exhausted, key B can still make all 3 requests."""
        headers_a = {"X-API-Key": TEST_API_KEY}
        headers_b = {"X-API-Key": SECOND_API_KEY}

        # Exhaust key A
        for i in range(4):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-isolate-a-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=headers_a
            )

        # Key B should work for all 3 requests
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-isolate-b-{i}"}
            resp = rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=headers_b
            )
            assert resp.status_code in (200, 201)


class TestRateLimitAuthInteraction:
    """Rate limiting interacts correctly with authentication."""

    def test_unauthenticated_requests_not_rate_limited(
        self, rate_limited_client: TestClient
    ) -> None:
        """Unauthenticated requests get 401, not 429 (auth before rate limit)."""
        for _ in range(5):
            resp = rate_limited_client.post("/api/v1/runs", json=VALID_PAYLOAD)
            assert resp.status_code == 401

    def test_invalid_key_not_rate_limited(
        self, rate_limited_client: TestClient
    ) -> None:
        """Invalid key requests always get 401, never 429."""
        headers = {"X-API-Key": "wrong-key"}
        for _ in range(5):
            resp = rate_limited_client.post(
                "/api/v1/runs", json=VALID_PAYLOAD, headers=headers
            )
            assert resp.status_code == 401


class TestRateLimitResponseSemantics:
    """429 response semantics are correct and informative."""

    def test_retry_after_is_positive_integer(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """``Retry-After`` value is a positive integer (seconds)."""
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-retry-val-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=auth_headers
            )

        payload = {**VALID_PAYLOAD, "report_id": "rl-retry-val-3"}
        resp = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=auth_headers
        )
        assert resp.status_code == 429
        retry_val = resp.headers["retry-after"]
        assert retry_val.isdigit()
        assert int(retry_val) >= 1

    def test_429_does_not_expose_internals(
        self, rate_limited_client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """429 response body does not expose internal details."""
        for i in range(3):
            payload = {**VALID_PAYLOAD, "report_id": f"rl-no-leak-{i}"}
            rate_limited_client.post(
                "/api/v1/runs", json=payload, headers=auth_headers
            )

        payload = {**VALID_PAYLOAD, "report_id": "rl-no-leak-3"}
        resp = rate_limited_client.post(
            "/api/v1/runs", json=payload, headers=auth_headers
        )
        body_str = str(resp.json())
        assert "traceback" not in body_str.lower()
        assert TEST_API_KEY not in body_str
