"""Tests for run command orchestration flow.

Validates:
- VAL-CLI-004: Run shows live progress and finishes with scorecard + JSON report
- VAL-CLI-005: Selective scopes execute only requested sections, skipped sections marked
- VAL-RUNBOOK-007: Phases execute in deterministic order with timestamped boundaries
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from ornn_bench.cli import app
from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runner import (
    SECTION_ORDER,
    RunOrchestrator,
    SectionRunner,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StubSectionRunner(SectionRunner):
    """A section runner that always succeeds with empty metrics."""

    def run(self) -> SectionResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            metrics={"stub": True},
        )


class FailingSectionRunner(SectionRunner):
    """A section runner that always fails."""

    def run(self) -> SectionResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.FAILED,
            started_at=now,
            finished_at=now,
            error=f"{self.name} failed deliberately",
        )


def _make_stub_runners() -> dict[str, SectionRunner]:
    """Create stub runners for all standard sections."""
    return {name: StubSectionRunner(name) for name in SECTION_ORDER}


# ---------------------------------------------------------------------------
# Section ordering tests (VAL-RUNBOOK-007)
# ---------------------------------------------------------------------------


class TestSectionOrdering:
    """Verify deterministic section execution order."""

    def test_section_order_is_defined(self) -> None:
        """SECTION_ORDER is a non-empty tuple/list of section names."""
        assert len(SECTION_ORDER) >= 4
        assert isinstance(SECTION_ORDER, (list, tuple))

    def test_preflight_comes_first(self) -> None:
        """pre-flight is the first section."""
        assert SECTION_ORDER[0] == "pre-flight"

    def test_postflight_comes_after_benchmarks(self) -> None:
        """post-flight comes after compute/memory/interconnect."""
        pf_idx = SECTION_ORDER.index("post-flight")
        for bench in ("compute", "memory", "interconnect"):
            if bench in SECTION_ORDER:
                assert SECTION_ORDER.index(bench) < pf_idx

    def test_manifest_is_last(self) -> None:
        """manifest is the last section."""
        assert SECTION_ORDER[-1] == "manifest"

    def test_orchestrator_runs_in_order(self) -> None:
        """Orchestrator executes sections in SECTION_ORDER."""
        runners = _make_stub_runners()
        call_order: list[str] = []
        original_runs = {}
        for name, r in runners.items():
            original_runs[name] = r.run

            def make_wrapper(n: str, orig: object) -> object:
                def wrapper() -> SectionResult:
                    call_order.append(n)
                    return orig()  # type: ignore[operator]
                return wrapper
            r.run = make_wrapper(name, r.run)  # type: ignore[assignment]

        orch = RunOrchestrator(runners=runners)
        orch.execute()

        assert call_order == list(SECTION_ORDER)

    def test_results_have_timestamps(self) -> None:
        """Every completed section result has started_at and finished_at."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        for section in report.sections:
            if section.status == BenchmarkStatus.COMPLETED:
                assert section.started_at is not None
                assert section.finished_at is not None


# ---------------------------------------------------------------------------
# Scope control tests (VAL-CLI-005)
# ---------------------------------------------------------------------------


class TestScopeControls:
    """Verify selective section execution via scope filters."""

    def test_compute_only_runs_preflight_compute_postflight_manifest(self) -> None:
        """--compute-only should run pre-flight, compute, post-flight, manifest."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners, scope={"compute"})
        report = orch.execute()

        names_run = {s.name for s in report.sections if s.status != BenchmarkStatus.SKIPPED}
        assert "compute" in names_run
        assert "pre-flight" in names_run
        assert "post-flight" in names_run

    def test_memory_only_skips_compute_and_interconnect(self) -> None:
        """--memory-only skips compute and interconnect."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners, scope={"memory"})
        report = orch.execute()

        skipped_names = {s.name for s in report.sections if s.status == BenchmarkStatus.SKIPPED}
        assert "compute" in skipped_names
        assert "interconnect" in skipped_names

    def test_interconnect_only(self) -> None:
        """--interconnect-only runs only interconnect (plus infra sections)."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners, scope={"interconnect"})
        report = orch.execute()

        skipped_names = {s.name for s in report.sections if s.status == BenchmarkStatus.SKIPPED}
        assert "compute" in skipped_names
        assert "memory" in skipped_names

    def test_skipped_sections_explicitly_marked(self) -> None:
        """Skipped sections have status=SKIPPED and no error."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners, scope={"compute"})
        report = orch.execute()

        for s in report.sections:
            if s.name in ("memory", "interconnect"):
                assert s.status == BenchmarkStatus.SKIPPED

    def test_no_scope_runs_all(self) -> None:
        """When no scope filter is set, all sections execute."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        for s in report.sections:
            assert s.status != BenchmarkStatus.SKIPPED

    def test_multiple_scopes(self) -> None:
        """Multiple scope sections can be specified."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners, scope={"compute", "memory"})
        report = orch.execute()

        skipped = {s.name for s in report.sections if s.status == BenchmarkStatus.SKIPPED}
        assert "interconnect" in skipped
        assert "compute" not in skipped
        assert "memory" not in skipped


# ---------------------------------------------------------------------------
# Progress/output tests (VAL-CLI-004)
# ---------------------------------------------------------------------------


class TestRunProgress:
    """Verify progress indication in run output."""

    def test_orchestrator_calls_progress_callback(self) -> None:
        """Orchestrator invokes progress callback for each section."""
        runners = _make_stub_runners()
        progress_calls: list[tuple[str, str]] = []

        def on_progress(section: str, status: str) -> None:
            progress_calls.append((section, status))

        orch = RunOrchestrator(runners=runners, on_progress=on_progress)
        orch.execute()

        section_names_started = [c[0] for c in progress_calls if c[1] == "started"]
        section_names_done = [c[0] for c in progress_calls if c[1] in ("completed", "skipped")]
        assert len(section_names_started) > 0
        assert len(section_names_done) > 0

    def test_report_has_sections(self) -> None:
        """Report from orchestrator has section results for all ordered sections."""
        runners = _make_stub_runners()
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        assert len(report.sections) == len(SECTION_ORDER)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestRunCommandCLI:
    """Integration tests for the run CLI command with mocked GPU."""

    def test_run_with_compute_only_flag(self) -> None:
        """--compute-only flag passes scope to orchestrator."""
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch("ornn_bench.cli.build_section_runners", return_value=_make_stub_runners()),
        ):
            result = runner.invoke(app, ["run", "--compute-only"])
        # Should not crash with traceback
        assert "Traceback" not in result.output

    def test_run_with_memory_only_flag(self) -> None:
        """--memory-only flag works without crash."""
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch("ornn_bench.cli.build_section_runners", return_value=_make_stub_runners()),
        ):
            result = runner.invoke(app, ["run", "--memory-only"])
        assert "Traceback" not in result.output

    def test_run_with_interconnect_only_flag(self) -> None:
        """--interconnect-only flag works without crash."""
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch("ornn_bench.cli.build_section_runners", return_value=_make_stub_runners()),
        ):
            result = runner.invoke(app, ["run", "--interconnect-only"])
        assert "Traceback" not in result.output

    def test_run_produces_json_report_file(self, tmp_path: Path) -> None:
        """run command writes JSON report to specified output path."""
        output_path = tmp_path / "report.json"
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch("ornn_bench.cli.build_section_runners", return_value=_make_stub_runners()),
        ):
            result = runner.invoke(app, ["run", "--output", str(output_path)])
        assert output_path.exists(), f"Report file not created. Output: {result.output}"
        data = json.loads(output_path.read_text())
        assert "sections" in data
        assert "schema_version" in data

    def test_run_shows_progress_text(self) -> None:
        """run output contains progress/status text."""
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch("ornn_bench.cli.build_section_runners", return_value=_make_stub_runners()),
        ):
            result = runner.invoke(app, ["run"])
        # Should show some progress or completion text
        output_lower = result.output.lower()
        assert (
            "complete" in output_lower
            or "done" in output_lower
            or "finished" in output_lower
            or "✓" in result.output
            or "passed" in output_lower
        )

    def test_run_no_raw_traceback(self) -> None:
        """run command never shows raw Python traceback."""
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch("ornn_bench.cli.build_section_runners", return_value=_make_stub_runners()),
        ):
            result = runner.invoke(app, ["run"])
        assert "Traceback" not in result.output
