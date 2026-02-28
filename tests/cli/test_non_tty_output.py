"""Tests for non-TTY and JSON output cleanliness.

Validates VAL-CLI-011: When output is piped or JSON mode is enabled,
output is machine-readable and free of ANSI/progress-bar artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from ornn_bench.cli import app
from ornn_bench.display import render_scorecard_plain
from ornn_bench.models import (
    Qualification,
    ScoreResult,
    ScoreStatus,
)

runner = CliRunner()

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_REPORT = FIXTURE_DIR / "sample_report.json"

# Regex to match ANSI escape codes
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _has_ansi(text: str) -> bool:
    """Check if text contains ANSI escape sequences."""
    return bool(ANSI_PATTERN.search(text))


class TestReportJsonOutput:
    """Tests for JSON output mode of the report command."""

    def test_json_flag_outputs_valid_json(self) -> None:
        """--json flag outputs valid JSON to stdout."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0, f"Output: {result.output}"
        parsed = json.loads(result.output)
        assert isinstance(parsed, dict)

    def test_json_output_contains_report_id(self) -> None:
        """JSON output includes report_id field."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["report_id"] == "test-report-fixture-001"

    def test_json_output_contains_scores(self) -> None:
        """JSON output includes scores section."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "scores" in parsed
        assert parsed["scores"]["ornn_i"] == 92.5
        assert parsed["scores"]["ornn_t"] == 88.3

    def test_json_output_contains_sections(self) -> None:
        """JSON output includes sections list."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "sections" in parsed
        assert len(parsed["sections"]) > 0

    def test_json_output_no_ansi(self) -> None:
        """JSON output has no ANSI escape codes."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0
        assert not _has_ansi(result.output)

    def test_json_output_no_progress_artifacts(self) -> None:
        """JSON output has no progress bar characters."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0
        # Should not contain Rich progress/spinner characters
        assert "━" not in result.output
        assert "╸" not in result.output


class TestReportPlainOutput:
    """Tests for plain text output mode of the report command."""

    def test_plain_flag_outputs_readable_text(self) -> None:
        """--plain flag outputs clean readable text."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--plain"])
        assert result.exit_code == 0, f"Output: {result.output}"
        assert len(result.output.strip()) > 0

    def test_plain_output_no_ansi(self) -> None:
        """Plain output has no ANSI escape codes."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--plain"])
        assert result.exit_code == 0
        assert not _has_ansi(result.output)

    def test_plain_output_contains_scores(self) -> None:
        """Plain output includes score values."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--plain"])
        assert result.exit_code == 0
        assert "92.5" in result.output
        assert "88.3" in result.output

    def test_plain_output_contains_qualification(self) -> None:
        """Plain output includes qualification."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--plain"])
        assert result.exit_code == 0
        assert "premium" in result.output.lower()

    def test_plain_output_contains_report_metadata(self) -> None:
        """Plain output includes report metadata."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--plain"])
        assert result.exit_code == 0
        assert "test-report-fixture-001" in result.output
        assert "1.0.0" in result.output

    def test_plain_output_no_box_drawing(self) -> None:
        """Plain output has no Rich box-drawing characters."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--plain"])
        assert result.exit_code == 0
        # Rich panels use these box-drawing characters
        box_chars = {"╭", "╮", "╰", "╯", "│", "─", "┌", "┐", "└", "┘", "├", "┤"}
        for char in box_chars:
            assert char not in result.output, f"Found box char '{char}' in plain output"


class TestRenderScorecardPlain:
    """Tests for the render_scorecard_plain utility function."""

    def test_valid_scores_render(self) -> None:
        """Valid scores render without ANSI codes."""
        scores = ScoreResult(
            ornn_i=92.5,
            ornn_t=88.3,
            qualification=Qualification.PREMIUM,
            components={"bw": 1.25, "fp8": 0.95, "bf16": 1.1, "ar": 0.85},
            score_status=ScoreStatus.VALID,
        )
        output = render_scorecard_plain(scores)
        assert not _has_ansi(output)
        assert "92.5" in output
        assert "88.3" in output
        assert "Premium" in output

    def test_partial_scores_render(self) -> None:
        """Partial scores render with N/A for missing values."""
        scores = ScoreResult(
            ornn_i=85.0,
            ornn_t=None,
            qualification=None,
            components={"bw": 1.0, "fp8": 0.8},
            score_status=ScoreStatus.PARTIAL,
            score_status_detail="Missing training metrics",
        )
        output = render_scorecard_plain(scores)
        assert not _has_ansi(output)
        assert "85.0" in output
        assert "N/A" in output
        assert "partial" in output.lower()

    def test_error_scores_render(self) -> None:
        """Error state scores render cleanly."""
        scores = ScoreResult(
            ornn_i=None,
            ornn_t=None,
            score_status=ScoreStatus.ERROR,
            score_status_detail="No valid metrics available",
        )
        output = render_scorecard_plain(scores)
        assert not _has_ansi(output)
        assert "N/A" in output
        assert "error" in output.lower()


class TestNonTTYOutputSafety:
    """Tests ensuring piped output is machine-safe."""

    def test_report_default_no_crash_in_non_tty(self) -> None:
        """Report command works in non-TTY context (CliRunner is non-TTY)."""
        # CliRunner by default simulates non-TTY
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT)])
        assert result.exit_code == 0

    def test_help_output_clean(self) -> None:
        """Help output is clean and parseable."""
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0
        assert "report" in result.output.lower()

    def test_version_output_clean(self) -> None:
        """Version output is clean single line."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        output = result.output.strip()
        assert "ornn-bench" in output
        assert "\n" not in output.strip()

    def test_json_output_parseable_after_pipe(self) -> None:
        """JSON output is valid JSON parseable by json.loads."""
        result = runner.invoke(app, ["report", str(SAMPLE_REPORT), "--json"])
        assert result.exit_code == 0
        # Must be parseable without exceptions
        data = json.loads(result.output)
        assert "report_id" in data
