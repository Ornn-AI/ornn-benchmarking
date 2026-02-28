"""Tests for partial-failure continuation semantics.

Validates VAL-CLI-006: If an individual benchmark fails or times out,
the CLI records failed/timeout status, continues remaining eligible sections,
avoids raw Python traceback output, and returns a defined non-zero
partial-failure exit code.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

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
    """A section runner that always succeeds."""

    def run(self) -> SectionResult:
        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            metrics={"stub": True},
        )


class FailingSectionRunner(SectionRunner):
    """A section runner that returns FAILED status."""

    def run(self) -> SectionResult:
        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.FAILED,
            started_at=now,
            finished_at=now,
            error=f"{self.name} benchmark tool crashed",
        )


class TimeoutSectionRunner(SectionRunner):
    """A section runner that returns TIMEOUT status."""

    def run(self) -> SectionResult:
        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.TIMEOUT,
            started_at=now,
            finished_at=now,
            error=f"{self.name} timed out after 300s",
        )


class ExceptionSectionRunner(SectionRunner):
    """A section runner that raises an unexpected exception."""

    def run(self) -> SectionResult:
        raise RuntimeError(f"Unexpected error in {self.name}")


def _make_runners_with_failure(fail_section: str, fail_type: str = "failed") -> (
    dict[str, SectionRunner]
):
    """Create runners where one section fails/times out."""
    runners: dict[str, SectionRunner] = {}
    for name in SECTION_ORDER:
        if name == fail_section:
            if fail_type == "timeout":
                runners[name] = TimeoutSectionRunner(name)
            elif fail_type == "exception":
                runners[name] = ExceptionSectionRunner(name)
            else:
                runners[name] = FailingSectionRunner(name)
        else:
            runners[name] = StubSectionRunner(name)
    return runners


def _make_all_stub_runners() -> dict[str, SectionRunner]:
    return {name: StubSectionRunner(name) for name in SECTION_ORDER}


# ---------------------------------------------------------------------------
# Core partial-failure tests
# ---------------------------------------------------------------------------


class TestPartialFailureContinuation:
    """Verify that failures in one section don't abort remaining sections."""

    def test_compute_failure_continues_memory_and_interconnect(self) -> None:
        """When compute fails, memory and interconnect still execute."""
        runners = _make_runners_with_failure("compute")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["compute"].status == BenchmarkStatus.FAILED
        assert section_map["memory"].status == BenchmarkStatus.COMPLETED
        assert section_map["interconnect"].status == BenchmarkStatus.COMPLETED

    def test_memory_failure_continues_interconnect(self) -> None:
        """When memory fails, interconnect still executes."""
        runners = _make_runners_with_failure("memory")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["memory"].status == BenchmarkStatus.FAILED
        assert section_map["interconnect"].status == BenchmarkStatus.COMPLETED

    def test_multiple_failures_still_completes_healthy_sections(self) -> None:
        """Multiple sections can fail; remaining healthy sections still complete."""
        runners: dict[str, SectionRunner] = {}
        for name in SECTION_ORDER:
            if name in ("compute", "interconnect"):
                runners[name] = FailingSectionRunner(name)
            else:
                runners[name] = StubSectionRunner(name)

        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["compute"].status == BenchmarkStatus.FAILED
        assert section_map["interconnect"].status == BenchmarkStatus.FAILED
        assert section_map["memory"].status == BenchmarkStatus.COMPLETED
        assert section_map["pre-flight"].status == BenchmarkStatus.COMPLETED
        assert section_map["post-flight"].status == BenchmarkStatus.COMPLETED

    def test_failed_section_records_error_message(self) -> None:
        """Failed sections carry a descriptive error message."""
        runners = _make_runners_with_failure("compute")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["compute"].error is not None
        assert len(section_map["compute"].error) > 0


# ---------------------------------------------------------------------------
# Timeout handling tests
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Verify timeout status is properly recorded."""

    def test_timeout_section_marked_as_timeout(self) -> None:
        """Timed-out section has status=TIMEOUT."""
        runners = _make_runners_with_failure("memory", fail_type="timeout")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["memory"].status == BenchmarkStatus.TIMEOUT

    def test_timeout_continues_remaining_sections(self) -> None:
        """After timeout, remaining sections still execute."""
        runners = _make_runners_with_failure("memory", fail_type="timeout")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["interconnect"].status == BenchmarkStatus.COMPLETED

    def test_timeout_error_message_present(self) -> None:
        """Timeout section carries error message with timing info."""
        runners = _make_runners_with_failure("compute", fail_type="timeout")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["compute"].error is not None
        assert "timed out" in section_map["compute"].error.lower()


# ---------------------------------------------------------------------------
# Exception safety tests
# ---------------------------------------------------------------------------


class TestExceptionSafety:
    """Verify that unexpected exceptions are caught and recorded."""

    def test_exception_in_runner_caught_as_failure(self) -> None:
        """An unexpected exception in a runner is caught and recorded as FAILED."""
        runners = _make_runners_with_failure("compute", fail_type="exception")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["compute"].status == BenchmarkStatus.FAILED
        assert section_map["compute"].error is not None

    def test_exception_does_not_abort_remaining(self) -> None:
        """Remaining sections execute after an exception in earlier section."""
        runners = _make_runners_with_failure("compute", fail_type="exception")
        orch = RunOrchestrator(runners=runners)
        report = orch.execute()

        section_map = {s.name: s for s in report.sections}
        assert section_map["memory"].status == BenchmarkStatus.COMPLETED
        assert section_map["interconnect"].status == BenchmarkStatus.COMPLETED


# ---------------------------------------------------------------------------
# has_failures / exit code tests
# ---------------------------------------------------------------------------


class TestFailureReporting:
    """Verify failure status reporting for exit code determination."""

    def test_has_failures_false_when_all_pass(self) -> None:
        """has_failures returns False when all sections pass."""
        runners = _make_all_stub_runners()
        orch = RunOrchestrator(runners=runners)
        orch.execute()
        assert orch.has_failures is False

    def test_has_failures_true_when_section_fails(self) -> None:
        """has_failures returns True when any section fails."""
        runners = _make_runners_with_failure("compute")
        orch = RunOrchestrator(runners=runners)
        orch.execute()
        assert orch.has_failures is True

    def test_has_failures_true_when_section_times_out(self) -> None:
        """has_failures returns True when any section times out."""
        runners = _make_runners_with_failure("memory", fail_type="timeout")
        orch = RunOrchestrator(runners=runners)
        orch.execute()
        assert orch.has_failures is True


# ---------------------------------------------------------------------------
# CLI integration: partial-failure exit code (VAL-CLI-006)
# ---------------------------------------------------------------------------


class TestPartialFailureCLI:
    """CLI integration tests for partial failure behavior."""

    def test_partial_failure_exit_code(self) -> None:
        """CLI returns non-zero exit code when a section fails."""
        from ornn_bench.cli import app

        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch(
                "ornn_bench.cli.build_section_runners",
                return_value=_make_runners_with_failure("compute"),
            ),
        ):
            result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_full_success_exit_code_zero(self) -> None:
        """CLI returns exit code 0 when all sections pass."""
        from ornn_bench.cli import app

        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch(
                "ornn_bench.cli.build_section_runners",
                return_value=_make_all_stub_runners(),
            ),
        ):
            result = runner.invoke(app, ["run"])
        assert result.exit_code == 0

    def test_partial_failure_no_traceback(self) -> None:
        """No raw Python traceback in output on partial failure."""
        from ornn_bench.cli import app

        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch(
                "ornn_bench.cli.build_section_runners",
                return_value=_make_runners_with_failure("compute"),
            ),
        ):
            result = runner.invoke(app, ["run"])
        assert "Traceback" not in result.output
        assert "raise " not in result.output

    def test_partial_failure_report_still_written(self, tmp_path: Path) -> None:
        """JSON report is written even when a section fails."""
        from ornn_bench.cli import app

        output_path = tmp_path / "report.json"
        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch(
                "ornn_bench.cli.build_section_runners",
                return_value=_make_runners_with_failure("compute"),
            ),
        ):
            result = runner.invoke(app, ["run", "--output", str(output_path)])
        assert output_path.exists(), f"Report file not created. Output: {result.output}"
        data = json.loads(output_path.read_text())
        # Failed section should be present in report
        section_statuses = {s["name"]: s["status"] for s in data["sections"]}
        assert section_statuses["compute"] == "failed"

    def test_partial_failure_shows_status_in_output(self) -> None:
        """CLI output indicates which sections failed."""
        from ornn_bench.cli import app

        with (
            patch("ornn_bench.cli.check_gpu_available", return_value=(True, "Found 1 GPU")),
            patch(
                "ornn_bench.cli.build_section_runners",
                return_value=_make_runners_with_failure("compute"),
            ),
        ):
            result = runner.invoke(app, ["run"])
        output_lower = result.output.lower()
        assert "fail" in output_lower or "✗" in result.output or "error" in output_lower
