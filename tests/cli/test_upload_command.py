"""Tests for the ``ornn-bench upload`` command.

Covers:
  - VAL-CLI-009: Upload command validates local report shape before network call
    and returns clear auth/network/validation messages with non-zero exit codes.
  - VAL-CROSS-002: Local vs server score consistency surfacing.
  - VAL-CROSS-003: Upload retry idempotency (CLI-side dedupe key + duplicate response).
  - VAL-CROSS-005: Schema version mismatch handling.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ornn_bench.api_client import (
    AuthenticationError,
    MetricComparison,
    NetworkError,
    RateLimitError,
    UploadResult,
    ValidationError,
    VerifyResult,
)
from ornn_bench.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_report_path(tmp_path: Path) -> Path:
    """Write a valid sample report to a temp file and return its path."""
    report_data = {
        "schema_version": "1.0.0",
        "report_id": "test-upload-001",
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
                "error": None,
            }
        ],
        "scores": {
            "ornn_i": 92.5,
            "ornn_t": 88.3,
            "qualification": "Premium",
            "components": {"bw": 1.25, "fp8": 0.95, "bf16": 1.1, "ar": 0.85},
            "score_status": "valid",
            "score_status_detail": None,
            "aggregate_method": None,
            "per_gpu_scores": [],
        },
        "manifest": {},
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report_data, indent=2))
    return path


@pytest.fixture()
def bad_schema_report_path(tmp_path: Path) -> Path:
    """Write a report with an unsupported schema version."""
    report_data = {
        "schema_version": "99.0.0",
        "report_id": "test-bad-schema",
        "created_at": "2024-01-15T10:30:00Z",
        "system_inventory": {"gpus": [], "os_info": "Ubuntu", "kernel_version": "5.15"},
        "sections": [],
        "scores": {
            "ornn_i": None,
            "ornn_t": None,
            "score_status": "error",
        },
        "manifest": {},
    }
    path = tmp_path / "bad_schema_report.json"
    path.write_text(json.dumps(report_data, indent=2))
    return path


@pytest.fixture()
def incomplete_report_path(tmp_path: Path) -> Path:
    """Write a report with missing required fields."""
    report_data = {
        "schema_version": "1.0.0",
        "report_id": "",
        "created_at": "",
        "system_inventory": {"gpus": []},
        "sections": [],
        "scores": {"score_status": "error"},
        "manifest": {},
    }
    path = tmp_path / "incomplete_report.json"
    path.write_text(json.dumps(report_data, indent=2))
    return path


# ---------------------------------------------------------------------------
# Tests: Missing prerequisites
# ---------------------------------------------------------------------------


class TestUploadPrerequisites:
    """Test upload prerequisites (API key, file existence, report shape)."""

    def test_missing_api_key_exits_nonzero(self, sample_report_path: Path) -> None:
        """Upload without API key exits with code 1 and shows guidance."""
        result = runner.invoke(app, ["upload", str(sample_report_path)])
        assert result.exit_code == 1
        assert "API key required" in result.output

    def test_missing_report_file_exits_nonzero(self, tmp_path: Path) -> None:
        """Upload with nonexistent file exits with code 1."""
        path = tmp_path / "nonexistent.json"
        result = runner.invoke(
            app, ["upload", str(path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_invalid_json_exits_nonzero(self, tmp_path: Path) -> None:
        """Upload with invalid JSON exits with code 1."""
        path = tmp_path / "bad.json"
        path.write_text("not valid json {{")
        result = runner.invoke(
            app, ["upload", str(path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "Failed to parse" in result.output

    def test_incomplete_report_validation_exits_nonzero(
        self, incomplete_report_path: Path
    ) -> None:
        """Upload with missing required fields exits with code 1 (VAL-CLI-009)."""
        result = runner.invoke(
            app, ["upload", str(incomplete_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "Validation Failed" in result.output


# ---------------------------------------------------------------------------
# Tests: Schema version mismatch (VAL-CROSS-005)
# ---------------------------------------------------------------------------


class TestUploadSchemaVersion:
    """Test schema version mismatch handling."""

    def test_unsupported_schema_version_exits_nonzero(
        self, bad_schema_report_path: Path
    ) -> None:
        """Upload with unsupported schema version exits with clear error."""
        result = runner.invoke(
            app, ["upload", str(bad_schema_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "schema version" in result.output.lower() or "Validation Failed" in result.output


# ---------------------------------------------------------------------------
# Tests: Error semantics (VAL-CLI-009)
# ---------------------------------------------------------------------------


class TestUploadErrorSemantics:
    """Test clear error messages for auth/network/validation failures."""

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_auth_error_shows_clear_message(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Authentication failure shows clear message with guidance."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.side_effect = AuthenticationError("Authentication failed: Unauthorized.")
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "bad-key"]
        )
        assert result.exit_code == 1
        assert "Authentication" in result.output

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_network_error_shows_retry_guidance(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Network failure shows retry guidance (safe to retry)."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.side_effect = NetworkError("Connection refused")
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "Network Error" in result.output
        assert "safe" in result.output.lower() or "retry" in result.output.lower()

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_validation_error_shows_details(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Server validation error shows details."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.side_effect = ValidationError(
            "Validation error: field 'report_id' is required"
        )
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "Validation" in result.output

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_rate_limit_error_shows_retry_after(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Rate limit error shows retry-after guidance."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.side_effect = RateLimitError(
            "Rate limit exceeded", retry_after=30
        )
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 1
        assert "Rate Limit" in result.output
        assert "30" in result.output


# ---------------------------------------------------------------------------
# Tests: Successful upload (VAL-CLI-009 success path)
# ---------------------------------------------------------------------------


class TestUploadSuccess:
    """Test successful upload behavior."""

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_successful_upload_shows_run_id(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Successful upload displays run ID and success message."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="abc123",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
            is_duplicate=False,
        )
        mock_client.verify.return_value = VerifyResult(
            status="verified",
            server_ornn_i=92.5,
            server_ornn_t=88.3,
            server_qualification="Premium",
            tolerance=0.01,
        )
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "Upload Complete" in result.output or "uploaded" in result.output.lower()

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_duplicate_upload_shows_existing_run_id(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Duplicate upload shows existing run ID (VAL-CROSS-003)."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="existing-run-id",
            received_at="2024-01-15T10:00:00Z",
            stored_at="2024-01-15T10:00:00Z",
            is_duplicate=True,
        )
        mock_client.verify.return_value = VerifyResult(
            status="verified",
            server_ornn_i=92.5,
            server_ornn_t=88.3,
            server_qualification="Premium",
            tolerance=0.01,
        )
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 0
        assert "existing-run-id" in result.output
        assert "Duplicate" in result.output or "already" in result.output.lower()

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_upload_with_no_verify(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Upload with --no-verify skips verification."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="xyz789",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
            is_duplicate=False,
        )
        result = runner.invoke(
            app,
            ["upload", str(sample_report_path), "--api-key", "test-key", "--no-verify"],
        )
        assert result.exit_code == 0
        assert "xyz789" in result.output
        # verify should not have been called
        mock_client.verify.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Score verification display (VAL-CROSS-002)
# ---------------------------------------------------------------------------


class TestUploadVerification:
    """Test local vs server score comparison display."""

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_verified_scores_show_match(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Matching scores show 'Verified' status."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="match-run",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
        )
        mock_client.verify.return_value = VerifyResult(
            status="verified",
            server_ornn_i=92.5,
            server_ornn_t=88.3,
            server_qualification="Premium",
            tolerance=0.01,
        )
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 0
        assert "Verified" in result.output or "match" in result.output.lower()

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_mismatched_scores_show_details(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Mismatched scores show per-metric details (VAL-CROSS-002)."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="mismatch-run",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
        )
        mock_client.verify.return_value = VerifyResult(
            status="mismatch",
            server_ornn_i=90.0,
            server_ornn_t=85.0,
            server_qualification="Premium",
            metric_details=[
                MetricComparison(
                    metric="ornn_i",
                    submitted=92.5,
                    server_computed=90.0,
                    match=False,
                    delta=2.5,
                ),
                MetricComparison(
                    metric="ornn_t",
                    submitted=88.3,
                    server_computed=85.0,
                    match=False,
                    delta=3.3,
                ),
                MetricComparison(
                    metric="qualification",
                    submitted=None,
                    server_computed=None,
                    match=True,
                ),
            ],
            tolerance=0.01,
        )
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        assert result.exit_code == 0
        assert "Mismatch" in result.output
        assert "ornn_i" in result.output
        assert "ornn_t" in result.output

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_verify_network_error_is_graceful(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """Verification network failure does not prevent upload success."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="ok-run",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
        )
        mock_client.verify.side_effect = NetworkError("timeout")
        result = runner.invoke(
            app, ["upload", str(sample_report_path), "--api-key", "test-key"]
        )
        # Upload succeeded even though verify failed
        assert result.exit_code == 0
        assert "ok-run" in result.output
        assert "skipped" in result.output.lower()


# ---------------------------------------------------------------------------
# Tests: API key via env var
# ---------------------------------------------------------------------------


class TestUploadEnvVar:
    """Test API key sourced from environment variable."""

    @patch("ornn_bench.cli.OrnnApiClient")
    def test_api_key_from_env_var(
        self, mock_client_cls: MagicMock, sample_report_path: Path
    ) -> None:
        """API key can be provided via ORNN_API_KEY env var."""
        mock_client = mock_client_cls.return_value
        mock_client.upload.return_value = UploadResult(
            run_id="env-run",
            received_at="2024-01-15T11:00:00Z",
            stored_at="2024-01-15T11:00:00Z",
        )
        mock_client.verify.return_value = VerifyResult(
            status="verified",
            server_ornn_i=92.5,
            server_ornn_t=88.3,
            server_qualification="Premium",
            tolerance=0.01,
        )
        result = runner.invoke(
            app,
            ["upload", str(sample_report_path)],
            env={"ORNN_API_KEY": "env-test-key"},
        )
        assert result.exit_code == 0
        assert "env-run" in result.output
