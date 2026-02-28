"""Tests for POST /api/v1/verify — score verification endpoint.

Covers:
  - VAL-API-005: Verification endpoint mismatch detail
  - VAL-API-009: Revoked key rejection (verify endpoint)
  - VAL-CROSS-002: Local vs server score consistency
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from api.auth import reset_api_keys
from fastapi.testclient import TestClient

from tests.api.conftest import TEST_API_KEY

# ---------------------------------------------------------------------------
# Helpers — deterministic fixture vectors
# ---------------------------------------------------------------------------

# Perfect match: components yield exactly the submitted scores
MATCH_PAYLOAD: dict[str, Any] = {
    "components": {"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
    "ornn_i": 100.0,  # 55*(1/1) + 45*(1/1) = 100
    "ornn_t": 100.0,  # 55*(1/1) + 45*(1/1) = 100
    "qualification": "Premium",
}

# Mismatch: submitted scores do not match server recomputation
MISMATCH_PAYLOAD: dict[str, Any] = {
    "components": {"bw": 2.0, "fp8": 0.5, "bf16": 1.0, "ar": 1.0},
    "ornn_i": 999.0,  # wrong — server computes 55*2 + 45*0.5 = 132.5
    "ornn_t": 100.0,  # correct — 55*1 + 45*1 = 100
    "qualification": "Premium",
}

# Partial components: missing some metrics
PARTIAL_PAYLOAD: dict[str, Any] = {
    "components": {"bw": 1.0, "fp8": 1.0},
    "ornn_i": 100.0,
    "ornn_t": None,
    "qualification": None,
}


# ---------------------------------------------------------------------------
# VAL-API-005: Verification endpoint — match case
# ---------------------------------------------------------------------------


class TestVerifyMatch:
    """Tests for verification when submitted scores match server recomputation."""

    def test_match_returns_200(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Matching scores return 200."""
        resp = client.post("/api/v1/verify", json=MATCH_PAYLOAD, headers=auth_headers)
        assert resp.status_code == 200

    def test_match_status_is_verified(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response status is 'verified' when all scores match."""
        data = client.post(
            "/api/v1/verify", json=MATCH_PAYLOAD, headers=auth_headers
        ).json()
        assert data["status"] == "verified"

    def test_match_server_scores_present(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response includes server-recomputed scores."""
        data = client.post(
            "/api/v1/verify", json=MATCH_PAYLOAD, headers=auth_headers
        ).json()
        assert data["server_ornn_i"] == pytest.approx(100.0)
        assert data["server_ornn_t"] == pytest.approx(100.0)
        assert data["server_qualification"] == "Premium"

    def test_match_metric_details_all_true(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """All metric_details have match=True."""
        data = client.post(
            "/api/v1/verify", json=MATCH_PAYLOAD, headers=auth_headers
        ).json()
        for detail in data["metric_details"]:
            assert detail["match"] is True

    def test_match_includes_tolerance(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response includes the tolerance used for comparison."""
        data = client.post(
            "/api/v1/verify", json=MATCH_PAYLOAD, headers=auth_headers
        ).json()
        assert "tolerance" in data
        assert isinstance(data["tolerance"], float)
        assert data["tolerance"] > 0

    def test_within_tolerance_still_verified(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Scores within tolerance (< 0.01) are still verified."""
        payload = {
            "components": {"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            "ornn_i": 100.005,  # delta = 0.005, within 0.01 tolerance
            "ornn_t": 99.995,   # delta = 0.005, within 0.01 tolerance
            "qualification": "Premium",
        }
        data = client.post(
            "/api/v1/verify", json=payload, headers=auth_headers
        ).json()
        assert data["status"] == "verified"


# ---------------------------------------------------------------------------
# VAL-API-005: Verification endpoint — mismatch case
# ---------------------------------------------------------------------------


class TestVerifyMismatch:
    """Tests for verification when submitted scores do NOT match server."""

    def test_mismatch_returns_200(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Mismatch still returns 200 (verification succeeded, values differ)."""
        resp = client.post(
            "/api/v1/verify", json=MISMATCH_PAYLOAD, headers=auth_headers
        )
        assert resp.status_code == 200

    def test_mismatch_status(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response status is 'mismatch'."""
        data = client.post(
            "/api/v1/verify", json=MISMATCH_PAYLOAD, headers=auth_headers
        ).json()
        assert data["status"] == "mismatch"

    def test_mismatch_server_scores(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Server recomputation values are present and correct."""
        data = client.post(
            "/api/v1/verify", json=MISMATCH_PAYLOAD, headers=auth_headers
        ).json()
        # bw=2.0, fp8=0.5 → 55*2 + 45*0.5 = 132.5
        assert data["server_ornn_i"] == pytest.approx(132.5)
        assert data["server_ornn_t"] == pytest.approx(100.0)

    def test_mismatch_per_metric_details(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Per-metric details show which metrics mismatched."""
        data = client.post(
            "/api/v1/verify", json=MISMATCH_PAYLOAD, headers=auth_headers
        ).json()
        details_by_metric = {d["metric"]: d for d in data["metric_details"]}

        # ornn_i should mismatch
        ornn_i_detail = details_by_metric["ornn_i"]
        assert ornn_i_detail["match"] is False
        assert ornn_i_detail["submitted"] == pytest.approx(999.0)
        assert ornn_i_detail["server_computed"] == pytest.approx(132.5)
        assert ornn_i_detail["delta"] is not None
        assert ornn_i_detail["delta"] > 0

        # ornn_t should match
        ornn_t_detail = details_by_metric["ornn_t"]
        assert ornn_t_detail["match"] is True

    def test_mismatch_delta_reported(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Delta is the absolute difference between submitted and server."""
        data = client.post(
            "/api/v1/verify", json=MISMATCH_PAYLOAD, headers=auth_headers
        ).json()
        details_by_metric = {d["metric"]: d for d in data["metric_details"]}
        ornn_i_detail = details_by_metric["ornn_i"]
        expected_delta = abs(999.0 - 132.5)
        assert ornn_i_detail["delta"] == pytest.approx(expected_delta)

    def test_qualification_mismatch_detected(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Qualification mismatch is detected when scores differ."""
        payload = {
            "components": {"bw": 0.5, "fp8": 0.5, "bf16": 0.5, "ar": 0.5},
            "ornn_i": 50.0,
            "ornn_t": 50.0,
            "qualification": "Premium",  # should be Below
        }
        data = client.post(
            "/api/v1/verify", json=payload, headers=auth_headers
        ).json()
        assert data["status"] == "mismatch"
        details_by_metric = {d["metric"]: d for d in data["metric_details"]}
        assert details_by_metric["qualification"]["match"] is False


# ---------------------------------------------------------------------------
# Partial components and edge cases
# ---------------------------------------------------------------------------


class TestVerifyEdgeCases:
    """Edge cases for the verify endpoint."""

    def test_partial_components_verified(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Partial components (only inference metrics) can still verify."""
        data = client.post(
            "/api/v1/verify", json=PARTIAL_PAYLOAD, headers=auth_headers
        ).json()
        assert data["status"] == "verified"
        assert data["server_ornn_i"] == pytest.approx(100.0)
        assert data["server_ornn_t"] is None

    def test_empty_components_verified_with_none_scores(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Empty components with None scores → verified (both are None)."""
        payload: dict[str, Any] = {
            "components": {},
            "ornn_i": None,
            "ornn_t": None,
            "qualification": None,
        }
        data = client.post(
            "/api/v1/verify", json=payload, headers=auth_headers
        ).json()
        assert data["status"] == "verified"

    def test_empty_components_with_nonzero_submitted_is_mismatch(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Empty components but non-None submitted scores → mismatch."""
        payload: dict[str, Any] = {
            "components": {},
            "ornn_i": 50.0,
            "ornn_t": 50.0,
            "qualification": "Standard",
        }
        data = client.post(
            "/api/v1/verify", json=payload, headers=auth_headers
        ).json()
        assert data["status"] == "mismatch"

    def test_missing_components_field_returns_422(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Missing required 'components' field returns 422."""
        payload: dict[str, Any] = {
            "ornn_i": 100.0,
            "ornn_t": 100.0,
        }
        resp = client.post("/api/v1/verify", json=payload, headers=auth_headers)
        assert resp.status_code == 422

    def test_response_content_type_json(
        self, client: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Response Content-Type is application/json."""
        resp = client.post(
            "/api/v1/verify", json=MATCH_PAYLOAD, headers=auth_headers
        )
        assert resp.headers["content-type"] == "application/json"


# ---------------------------------------------------------------------------
# Auth and rate limiting on verify endpoint
# ---------------------------------------------------------------------------


class TestVerifyAuth:
    """Auth checks on POST /api/v1/verify."""

    def test_missing_api_key_returns_401(self, client: TestClient) -> None:
        """Missing API key returns 401."""
        resp = client.post("/api/v1/verify", json=MATCH_PAYLOAD)
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, client: TestClient) -> None:
        """Invalid API key returns 401."""
        headers = {"X-API-Key": "wrong-key"}
        resp = client.post("/api/v1/verify", json=MATCH_PAYLOAD, headers=headers)
        assert resp.status_code == 401

    def test_revoked_key_returns_401(self, client: TestClient) -> None:
        """Revoked API key returns 401 on verify endpoint (VAL-API-009)."""
        revoked_key = "revoked-verify-key"
        reset_api_keys()
        os.environ["ORNN_API_KEYS"] = f"{TEST_API_KEY},{revoked_key}"
        os.environ["ORNN_REVOKED_API_KEYS"] = revoked_key
        reset_api_keys()

        headers = {"X-API-Key": revoked_key}
        resp = client.post("/api/v1/verify", json=MATCH_PAYLOAD, headers=headers)
        assert resp.status_code == 401

        # Clean up
        os.environ["ORNN_API_KEYS"] = TEST_API_KEY
        os.environ.pop("ORNN_REVOKED_API_KEYS", None)
        reset_api_keys()

    def test_401_no_sensitive_details(self, client: TestClient) -> None:
        """401 response does not leak key values."""
        resp = client.post("/api/v1/verify", json=MATCH_PAYLOAD)
        body = resp.json()
        assert TEST_API_KEY not in str(body)
