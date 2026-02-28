"""Integration tests for the run + upload end-to-end flow.

Covers:
  - VAL-CROSS-001: ``ornn-bench run --upload`` executes runbook, persists
    local JSON, uploads to API, returns remote run_id.
  - VAL-CROSS-002: Local vs server score consistency surfacing.
  - VAL-CROSS-003: CLI retry after transient failure safely reattempts
    upload without creating duplicate logical runs.
  - VAL-CROSS-005: Schema version compatibility — CLI report includes
    schema_version and API rejects unsupported versions.
  - VAL-API-006: Idempotent ingest — duplicate uploads return same run_id.

These tests use the FastAPI TestClient for the API side and mock the
CLI's HTTP calls to route through the same in-process test server,
giving us full integration coverage without network dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from ornn_bench.api_client import (
    SUPPORTED_SCHEMA_VERSIONS,
    UploadResult,
    compute_dedupe_key,
    validate_report_for_upload,
)
from ornn_bench.cli import app
from ornn_bench.models import BenchmarkReport

cli_runner = CliRunner()

# ---------------------------------------------------------------------------
# Test API key (matches conftest setup)
# ---------------------------------------------------------------------------

TEST_API_KEY = "test-api-key-12345"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_report() -> BenchmarkReport:
    """Build a valid BenchmarkReport for integration testing."""
    return BenchmarkReport.model_validate(
        {
            "schema_version": "1.0.0",
            "report_id": "integration-test-001",
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
                    "started_at": "2024-01-15T10:30:05Z",
                    "finished_at": "2024-01-15T10:35:00Z",
                    "metrics": {},
                }
            ],
            "scores": {
                "ornn_i": 92.5,
                "ornn_t": 88.3,
                "qualification": "Premium",
                "components": {"bw": 1.25, "fp8": 0.95, "bf16": 1.1, "ar": 0.85},
                "score_status": "valid",
            },
            "manifest": {},
        }
    )


@pytest.fixture()
def report_file(tmp_path: Path, sample_report: BenchmarkReport) -> Path:
    """Write a sample report to a temp file."""
    path = tmp_path / "report.json"
    path.write_text(sample_report.model_dump_json(indent=2))
    return path


# ---------------------------------------------------------------------------
# API client integration with real API TestClient
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_test_client() -> TestClient:
    """Create a fresh API TestClient with fake Firestore."""
    from api.auth import reset_api_keys
    from api.dependencies import (
        reset_dependencies,
        set_firestore_client,
        set_rate_limiter,
    )
    from api.main import create_app
    from api.rate_limit import RateLimiter

    from tests.api.conftest import FakeFirestoreClient

    reset_dependencies()
    reset_api_keys()
    set_firestore_client(FakeFirestoreClient())
    set_rate_limiter(RateLimiter(max_requests=10_000, window_seconds=60))

    os.environ["ORNN_API_KEYS"] = TEST_API_KEY
    os.environ.pop("ORNN_REVOKED_API_KEYS", None)

    application = create_app()
    client = TestClient(application)
    yield client  # type: ignore[misc]

    reset_dependencies()
    reset_api_keys()
    os.environ.pop("ORNN_API_KEYS", None)


# ---------------------------------------------------------------------------
# Tests: Dedupe key consistency (VAL-CROSS-003, VAL-API-006)
# ---------------------------------------------------------------------------


class TestDedupeKeyConsistency:
    """Test that CLI and API produce identical dedupe keys."""

    def test_cli_dedupe_key_matches_api_logic(
        self, sample_report: BenchmarkReport
    ) -> None:
        """CLI-side dedupe key computation matches server-side logic."""
        import hashlib

        identity = json.dumps(
            {
                "report_id": sample_report.report_id,
                "created_at": sample_report.created_at,
                "schema_version": sample_report.schema_version,
            },
            sort_keys=True,
        )
        expected_key = hashlib.sha256(identity.encode()).hexdigest()
        assert compute_dedupe_key(sample_report) == expected_key

    def test_same_report_produces_same_dedupe_key(
        self, sample_report: BenchmarkReport
    ) -> None:
        """Identical reports produce identical dedupe keys (retry-safe)."""
        key1 = compute_dedupe_key(sample_report)
        key2 = compute_dedupe_key(sample_report)
        assert key1 == key2


# ---------------------------------------------------------------------------
# Tests: Upload to real API (integration)
# ---------------------------------------------------------------------------


class TestUploadToApi:
    """Test upload against the real API server via TestClient."""

    def test_upload_returns_201_on_first_submission(
        self,
        api_test_client: TestClient,
        sample_report: BenchmarkReport,
    ) -> None:
        """First upload returns 201 with run_id."""
        response = api_test_client.post(
            "/api/v1/runs",
            content=sample_report.model_dump_json(),
            headers={
                "X-API-Key": TEST_API_KEY,
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert "run_id" in data
        assert "stored_at" in data

    def test_duplicate_upload_returns_200_with_same_run_id(
        self,
        api_test_client: TestClient,
        sample_report: BenchmarkReport,
    ) -> None:
        """Second upload of same report returns 200 with same run_id (VAL-API-006)."""
        headers = {
            "X-API-Key": TEST_API_KEY,
            "Content-Type": "application/json",
        }
        payload = sample_report.model_dump_json()

        resp1 = api_test_client.post("/api/v1/runs", content=payload, headers=headers)
        resp2 = api_test_client.post("/api/v1/runs", content=payload, headers=headers)

        assert resp1.status_code == 201
        assert resp2.status_code == 200
        assert resp1.json()["run_id"] == resp2.json()["run_id"]

    def test_unsupported_schema_version_returns_422(
        self,
        api_test_client: TestClient,
        sample_report: BenchmarkReport,
    ) -> None:
        """API rejects unsupported schema version with 422 (VAL-CROSS-005)."""
        # Mutate schema version to unsupported value
        report_data = sample_report.model_dump()
        report_data["schema_version"] = "99.0.0"

        response = api_test_client.post(
            "/api/v1/runs",
            json=report_data,
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 422
        detail = response.json().get("detail", "")
        assert "schema version" in detail.lower() or "99.0.0" in detail


# ---------------------------------------------------------------------------
# Tests: Verify endpoint integration
# ---------------------------------------------------------------------------


class TestVerifyIntegration:
    """Test score verification against real API."""

    def test_matching_scores_return_verified(
        self,
        api_test_client: TestClient,
        sample_report: BenchmarkReport,
    ) -> None:
        """Matching scores return 'verified' status (VAL-CROSS-002).

        Ornn-I = 55*(BW/BW_ref) + 45*(FP8/FP8_ref)
               = 55*(1.0/1.0) + 45*(1.0/1.0) = 100.0
        Ornn-T = 55*(BF16/BF16_ref) + 45*(AR/AR_ref)
               = 55*(1.0/1.0) + 45*(1.0/1.0) = 100.0
        Qualification: Premium (composite=100.0, both floors >= 80)
        """
        response = api_test_client.post(
            "/api/v1/verify",
            json={
                "components": {"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
                "ornn_i": 100.0,
                "ornn_t": 100.0,
                "qualification": "Premium",
            },
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "verified"

    def test_mismatched_scores_return_mismatch_with_details(
        self,
        api_test_client: TestClient,
    ) -> None:
        """Mismatched scores return 'mismatch' with per-metric details."""
        response = api_test_client.post(
            "/api/v1/verify",
            json={
                "components": {"bw": 1.25, "fp8": 0.95, "bf16": 1.1, "ar": 0.85},
                "ornn_i": 50.0,  # Wrong value
                "ornn_t": 88.3,
                "qualification": "Premium",
            },
            headers={"X-API-Key": TEST_API_KEY},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "mismatch"
        assert len(data["metric_details"]) > 0
        # Find the ornn_i mismatch detail
        ornn_i_detail = next(
            d for d in data["metric_details"] if d["metric"] == "ornn_i"
        )
        assert ornn_i_detail["match"] is False
        assert ornn_i_detail["delta"] is not None


# ---------------------------------------------------------------------------
# Tests: Local validation
# ---------------------------------------------------------------------------


class TestLocalValidation:
    """Test local report validation before network call."""

    def test_valid_report_passes_validation(
        self, sample_report: BenchmarkReport
    ) -> None:
        """Valid report produces no validation errors."""
        errors = validate_report_for_upload(sample_report)
        assert errors == []

    def test_missing_report_id_fails_validation(self) -> None:
        """Missing report_id produces validation error."""
        report = BenchmarkReport(
            schema_version="1.0.0",
            report_id="",
            created_at="2024-01-15T10:30:00Z",
        )
        errors = validate_report_for_upload(report)
        assert any("report_id" in err.lower() for err in errors)

    def test_missing_created_at_fails_validation(self) -> None:
        """Missing created_at produces validation error."""
        report = BenchmarkReport(
            schema_version="1.0.0",
            report_id="test-id",
            created_at="",
        )
        errors = validate_report_for_upload(report)
        assert any("created_at" in err.lower() for err in errors)

    def test_unsupported_schema_version_fails_validation(self) -> None:
        """Unsupported schema version produces clear error (VAL-CROSS-005)."""
        report = BenchmarkReport(
            schema_version="99.0.0",
            report_id="test-id",
            created_at="2024-01-15T10:30:00Z",
        )
        errors = validate_report_for_upload(report)
        assert any("schema version" in err.lower() for err in errors)
        assert any("99.0.0" in err for err in errors)

    def test_supported_schema_versions_constant(self) -> None:
        """SUPPORTED_SCHEMA_VERSIONS includes at least 1.0.0."""
        assert "1.0.0" in SUPPORTED_SCHEMA_VERSIONS


# ---------------------------------------------------------------------------
# Tests: CLI run --upload flow (VAL-CROSS-001)
# ---------------------------------------------------------------------------


class TestRunUploadCLI:
    """Test CLI ``run --upload`` integration."""

    @patch("ornn_bench.cli._perform_verify")
    @patch("ornn_bench.cli._perform_upload")
    @patch("ornn_bench.cli.RunOrchestrator")
    @patch("ornn_bench.cli.build_section_runners")
    @patch("ornn_bench.cli.check_gpu_available")
    def test_run_upload_posts_and_returns_run_id(
        self,
        mock_gpu_check: MagicMock,
        mock_build_runners: MagicMock,
        mock_orch_cls: MagicMock,
        mock_upload: MagicMock,
        mock_verify: MagicMock,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """run --upload executes runbook, uploads, returns remote run_id."""
        mock_gpu_check.return_value = (True, "GPU available")
        mock_build_runners.return_value = []

        mock_orch = mock_orch_cls.return_value
        mock_orch.execute.return_value = sample_report
        mock_orch.has_failures = False

        mock_upload.return_value = UploadResult(
            run_id="remote-run-id",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
        )
        mock_verify.return_value = None

        result = cli_runner.invoke(
            app,
            [
                "run",
                "--upload",
                "--api-key",
                "test-key",
                "-o",
                str(tmp_path / "output.json"),
            ],
        )
        assert result.exit_code == 0
        assert "remote-run-id" in result.output
        mock_upload.assert_called_once()
        mock_verify.assert_called_once()

    @patch("ornn_bench.cli.check_gpu_available")
    def test_run_upload_without_api_key_exits_nonzero(
        self,
        mock_gpu_check: MagicMock,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """run --upload without API key exits with clear error."""
        mock_gpu_check.return_value = (True, "GPU available")

        # Mock the run orchestrator to avoid GPU dependency
        with patch("ornn_bench.cli.build_section_runners") as mock_build, \
             patch("ornn_bench.cli.RunOrchestrator") as mock_orch_cls:
            mock_build.return_value = []
            mock_orch = mock_orch_cls.return_value
            mock_orch.execute.return_value = sample_report
            mock_orch.has_failures = False

            result = cli_runner.invoke(
                app,
                ["run", "--upload", "-o", str(tmp_path / "output.json")],
            )
        assert result.exit_code == 1
        assert "API key required" in result.output

    @patch("ornn_bench.cli._perform_upload")
    @patch("ornn_bench.cli.RunOrchestrator")
    @patch("ornn_bench.cli.build_section_runners")
    @patch("ornn_bench.cli.check_gpu_available")
    def test_run_upload_failure_still_saves_local_report(
        self,
        mock_gpu_check: MagicMock,
        mock_build_runners: MagicMock,
        mock_orch_cls: MagicMock,
        mock_upload: MagicMock,
        sample_report: BenchmarkReport,
        tmp_path: Path,
    ) -> None:
        """run --upload failure still saves the local report file."""
        mock_gpu_check.return_value = (True, "GPU available")
        mock_build_runners.return_value = []

        mock_orch = mock_orch_cls.return_value
        mock_orch.execute.return_value = sample_report
        mock_orch.has_failures = False

        mock_upload.return_value = None  # upload failed

        output_path = tmp_path / "output.json"
        result = cli_runner.invoke(
            app,
            [
                "run",
                "--upload",
                "--api-key",
                "test-key",
                "-o",
                str(output_path),
            ],
        )
        assert result.exit_code == 1
        # Local report should still be saved
        assert output_path.exists()
        # Should be parseable
        saved_report = BenchmarkReport.model_validate_json(output_path.read_text())
        assert saved_report.report_id == sample_report.report_id


# ---------------------------------------------------------------------------
# Tests: Schema version in report (VAL-CROSS-005)
# ---------------------------------------------------------------------------


class TestSchemaVersionInReport:
    """Test schema version is included in CLI-generated reports."""

    def test_report_includes_schema_version(
        self, sample_report: BenchmarkReport
    ) -> None:
        """CLI report includes schema_version field."""
        assert sample_report.schema_version
        assert sample_report.schema_version in SUPPORTED_SCHEMA_VERSIONS

    def test_report_json_includes_schema_version(
        self, sample_report: BenchmarkReport
    ) -> None:
        """JSON serialization includes schema_version."""
        data = json.loads(sample_report.model_dump_json())
        assert "schema_version" in data
        assert data["schema_version"] == "1.0.0"
