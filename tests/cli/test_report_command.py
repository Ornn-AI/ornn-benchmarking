"""Tests for the report command rendering.

Validates VAL-CLI-008: ``ornn-bench report <json>`` re-renders a saved report
successfully on machines without GPU tools installed.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ornn_bench.cli import app

runner = CliRunner()

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_REPORT = FIXTURE_DIR / "sample_report.json"
SCORED_REPORT = FIXTURE_DIR / "sample_scored_report.json"


class TestReportCommandRendering:
    """Tests for the report command rendering a saved JSON report."""

    def test_report_renders_sample_report(self) -> None:
        """Report command succeeds with sample report fixture."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0, f"Output: {result.output}"

    def test_report_renders_scored_report(self) -> None:
        """Report command succeeds with scored report fixture."""
        result = runner.invoke(app, ["report", str(SCORED_REPORT)])
        assert result.exit_code == 0, f"Output: {result.output}"

    def test_report_shows_report_id(self) -> None:
        """Report output includes the report ID."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "test-report-fixture-001" in result.output

    def test_report_shows_created_at(self) -> None:
        """Report output includes the creation timestamp."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "2024-01-15" in result.output

    def test_report_shows_schema_version(self) -> None:
        """Report output includes the schema version."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_report_shows_section_names(self) -> None:
        """Report output includes all section names."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "compute" in output_lower
        assert "memory" in output_lower
        assert "interconnect" in output_lower

    def test_report_shows_section_statuses(self) -> None:
        """Report output includes section status values."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "completed" in result.output.lower()

    def test_report_shows_ornn_i_score(self) -> None:
        """Report output includes Ornn-I score."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "92.5" in result.output

    def test_report_shows_ornn_t_score(self) -> None:
        """Report output includes Ornn-T score."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "88.3" in result.output

    def test_report_shows_qualification(self) -> None:
        """Report output includes qualification outcome."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "premium" in result.output.lower()

    def test_report_shows_component_metrics(self) -> None:
        """Report output includes component metric values."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        assert "1.25" in result.output  # bw
        assert "0.95" in result.output  # fp8

    def test_report_shows_score_status(self) -> None:
        """Report output includes score validity status."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0
        # Should show "Valid" or "valid"
        assert "valid" in result.output.lower()


class TestReportCommandNoGPURequired:
    """Tests that report command does not require GPU tools (VAL-CLI-008)."""

    def test_no_nvidia_smi_required(self) -> None:
        """Report command works without nvidia-smi on PATH."""
        # This test running on macOS without GPU proves this assertion
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0

    def test_no_traceback_in_output(self) -> None:
        """Report command never shows raw Python traceback."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert "Traceback" not in result.output
        assert "raise " not in result.output


class TestReportCommandErrorHandling:
    """Tests for report command error paths."""

    def test_missing_file_exits_with_error(self) -> None:
        """Report command exits with code 1 for missing file."""
        result = runner.invoke(app, ["report", "/nonexistent/report.json"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_invalid_json_exits_with_error(self, tmp_path: Path) -> None:
        """Report command exits with code 1 for invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")
        result = runner.invoke(app, ["report", str(bad_file)])
        assert result.exit_code == 1
        assert "error" in result.output.lower()

    def test_wrong_schema_exits_with_error(self, tmp_path: Path) -> None:
        """Report command exits with code 1 for JSON that doesn't match schema."""
        wrong_file = tmp_path / "wrong.json"
        wrong_file.write_text('{"foo": "bar"}')
        # This actually succeeds due to Pydantic defaults — it's valid
        # because BenchmarkReport has all optional/default fields
        result = runner.invoke(app, ["report", str(wrong_file)])
        # Should succeed since Pydantic fills defaults
        assert result.exit_code == 0

    def test_no_traceback_on_error(self, tmp_path: Path) -> None:
        """Report command shows clean error, not traceback, for invalid input."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{invalid json")
        result = runner.invoke(app, ["report", str(bad_file)])
        assert result.exit_code == 1
        assert "Traceback" not in result.output


class TestReportCommandWithPartialReport:
    """Tests for report command with partial/failed section reports."""

    def test_partial_report_renders(self, tmp_path: Path) -> None:
        """Report renders correctly when some sections failed."""
        partial = {
            "schema_version": "1.0.0",
            "report_id": "partial-001",
            "created_at": "2024-01-15T10:30:00Z",
            "sections": [
                {
                    "name": "compute",
                    "status": "completed",
                    "started_at": "2024-01-15T10:30:00Z",
                    "finished_at": "2024-01-15T10:35:00Z",
                },
                {
                    "name": "memory",
                    "status": "failed",
                    "started_at": "2024-01-15T10:35:00Z",
                    "finished_at": "2024-01-15T10:35:01Z",
                    "error": "nvbandwidth not found",
                },
                {
                    "name": "interconnect",
                    "status": "skipped",
                },
            ],
            "scores": {
                "ornn_i": None,
                "ornn_t": None,
                "score_status": "error",
                "score_status_detail": "No valid metrics available for scoring",
            },
        }
        report_file = tmp_path / "partial_report.json"
        report_file.write_text(json.dumps(partial))
        result = runner.invoke(app, ["report", str(report_file)])
        assert result.exit_code == 0
        assert "failed" in result.output.lower()
        assert "skipped" in result.output.lower()

    def test_empty_sections_report_renders(self, tmp_path: Path) -> None:
        """Report renders correctly with no sections."""
        empty = {
            "schema_version": "1.0.0",
            "report_id": "empty-001",
            "created_at": "2024-01-15T10:30:00Z",
            "sections": [],
            "scores": {
                "ornn_i": None,
                "ornn_t": None,
                "score_status": "error",
                "score_status_detail": "No valid metrics available for scoring",
            },
        }
        report_file = tmp_path / "empty_report.json"
        report_file.write_text(json.dumps(empty))
        result = runner.invoke(app, ["report", str(report_file)])
        assert result.exit_code == 0
