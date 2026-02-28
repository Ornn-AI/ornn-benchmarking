"""Tests for manifest completeness tracking and interruption-safe persistence.

Validates:
- VAL-RUNBOOK-008: Manifest completeness and explicit omissions
  Output manifest enumerates expected artifacts for all runbook sections;
  any missing artifact includes explicit skip/failure reason.

- VAL-RUNBOOK-009: Partial-result durability
  If a run is interrupted or a later phase fails, previously completed
  phase results are still persisted in JSON report with terminal status.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runbook.manifest import (
    ManifestBuilder,
    ManifestRunner,
    ManifestStatus,
)
from ornn_bench.runner import (
    DurableRunOrchestrator,
    SectionRunner,
    StubSectionRunner,
)

# ---------------------------------------------------------------------------
# Helper runners for testing
# ---------------------------------------------------------------------------


class FailingSectionRunner(SectionRunner):
    """Runner that always raises an exception."""

    def run(self) -> SectionResult:
        raise RuntimeError("Simulated benchmark failure")


class InterruptingSectionRunner(SectionRunner):
    """Runner that raises KeyboardInterrupt to simulate interruption."""

    def run(self) -> SectionResult:
        raise KeyboardInterrupt("Simulated interruption")


# ---------------------------------------------------------------------------
# ManifestBuilder tests
# ---------------------------------------------------------------------------


class TestManifestBuilder:
    """Tests for the manifest builder."""

    def test_records_produced_artifact(self) -> None:
        """VAL-RUNBOOK-008: Produced artifacts are recorded."""
        builder = ManifestBuilder()
        builder.record_produced("pre-flight", "nvidia_smi_q_output")
        entries = builder.get_entries()
        assert len(entries) == 1
        assert entries[0].section == "pre-flight"
        assert entries[0].artifact == "nvidia_smi_q_output"
        assert entries[0].status == ManifestStatus.PRODUCED

    def test_records_skipped_artifact(self) -> None:
        """VAL-RUNBOOK-008: Skipped artifacts include reason."""
        builder = ManifestBuilder()
        builder.record_skipped("compute", "tf32_results", reason="TF32 not enabled")
        entries = builder.get_entries()
        assert len(entries) == 1
        assert entries[0].status == ManifestStatus.SKIPPED
        assert entries[0].reason == "TF32 not enabled"

    def test_records_missing_artifact(self) -> None:
        """VAL-RUNBOOK-008: Missing artifacts include failure reason."""
        builder = ManifestBuilder()
        builder.record_missing(
            "memory", "nvbandwidth_d2d_read", reason="nvbandwidth not found"
        )
        entries = builder.get_entries()
        assert len(entries) == 1
        assert entries[0].status == ManifestStatus.MISSING
        assert entries[0].reason == "nvbandwidth not found"

    def test_multiple_artifacts_tracked(self) -> None:
        builder = ManifestBuilder()
        builder.record_produced("pre-flight", "gpu_inventory")
        builder.record_produced("compute", "bf16_results")
        builder.record_skipped("compute", "tf32_results", reason="disabled")
        builder.record_missing("memory", "d2d_read", reason="tool missing")
        entries = builder.get_entries()
        assert len(entries) == 4

    def test_to_dict_format(self) -> None:
        """VAL-RUNBOOK-008: Manifest serializable to dict."""
        builder = ManifestBuilder()
        builder.record_produced("pre-flight", "gpu_inventory")
        builder.record_skipped("compute", "tf32", reason="disabled")
        manifest = builder.to_dict()
        assert "entries" in manifest
        assert "summary" in manifest
        assert manifest["summary"]["produced"] == 1
        assert manifest["summary"]["skipped"] == 1
        assert manifest["summary"]["missing"] == 0

    def test_build_from_sections(self) -> None:
        """VAL-RUNBOOK-008: Build manifest from section results."""
        sections = [
            SectionResult(
                name="pre-flight",
                status=BenchmarkStatus.COMPLETED,
                started_at="2024-01-01T00:00:00Z",
                finished_at="2024-01-01T00:01:00Z",
                metrics={"gpu_inventory": {"attached_gpus": 2}},
            ),
            SectionResult(
                name="compute",
                status=BenchmarkStatus.COMPLETED,
                started_at="2024-01-01T00:01:00Z",
                finished_at="2024-01-01T00:10:00Z",
                metrics={"dtypes_tested": ["bf16", "fp8_e4m3"]},
            ),
            SectionResult(
                name="memory",
                status=BenchmarkStatus.SKIPPED,
            ),
            SectionResult(
                name="interconnect",
                status=BenchmarkStatus.FAILED,
                error="nccl-tests not found",
            ),
        ]
        builder = ManifestBuilder()
        builder.build_from_sections(sections)
        manifest = builder.to_dict()
        # Should have entries for each section
        assert manifest["summary"]["produced"] >= 1
        # Skipped sections should be recorded
        skipped = [e for e in manifest["entries"] if e["status"] == "skipped"]
        assert len(skipped) > 0
        # Failed sections should be recorded as missing
        missing = [e for e in manifest["entries"] if e["status"] == "missing"]
        assert len(missing) > 0


# ---------------------------------------------------------------------------
# ManifestRunner tests
# ---------------------------------------------------------------------------


class TestManifestRunner:
    """Tests for the ManifestRunner section runner."""

    def test_runner_returns_completed(self) -> None:
        sections = [
            SectionResult(
                name="pre-flight",
                status=BenchmarkStatus.COMPLETED,
                started_at="2024-01-01T00:00:00Z",
                finished_at="2024-01-01T00:01:00Z",
            ),
        ]
        runner = ManifestRunner(sections=sections)
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_manifest_metrics(self) -> None:
        sections = [
            SectionResult(
                name="pre-flight",
                status=BenchmarkStatus.COMPLETED,
            ),
            SectionResult(
                name="compute",
                status=BenchmarkStatus.FAILED,
                error="mamf-finder not found",
            ),
        ]
        runner = ManifestRunner(sections=sections)
        result = runner.run()
        assert "entries" in result.metrics
        assert "summary" in result.metrics

    def test_runner_name_is_manifest(self) -> None:
        runner = ManifestRunner(sections=[])
        result = runner.run()
        assert result.name == "manifest"


# ---------------------------------------------------------------------------
# Durability tests (VAL-RUNBOOK-009)
# ---------------------------------------------------------------------------


class TestDurableRunOrchestrator:
    """Tests for interruption-safe report persistence."""

    def test_persists_on_normal_completion(self) -> None:
        """VAL-RUNBOOK-009: Report persisted on successful run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            runners = {
                "pre-flight": StubSectionRunner("pre-flight"),
                "compute": StubSectionRunner("compute"),
                "memory": StubSectionRunner("memory"),
                "interconnect": StubSectionRunner("interconnect"),
                "post-flight": StubSectionRunner("post-flight"),
                "manifest": StubSectionRunner("manifest"),
            }
            orch = DurableRunOrchestrator(
                runners=runners,
                output_path=output_path,
            )
            report = orch.execute()
            assert output_path.exists()
            # Verify persisted report is valid JSON
            data = json.loads(output_path.read_text())
            assert data["report_id"] == report.report_id

    def test_persists_completed_sections_on_failure(self) -> None:
        """VAL-RUNBOOK-009: Completed sections persisted when later section fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            runners = {
                "pre-flight": StubSectionRunner("pre-flight"),
                "compute": StubSectionRunner("compute"),
                "memory": FailingSectionRunner("memory"),
                "interconnect": StubSectionRunner("interconnect"),
                "post-flight": StubSectionRunner("post-flight"),
                "manifest": StubSectionRunner("manifest"),
            }
            orch = DurableRunOrchestrator(
                runners=runners,
                output_path=output_path,
            )
            orch.execute()
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            sections = {s["name"]: s for s in data["sections"]}
            # Pre-flight and compute should be completed
            assert sections["pre-flight"]["status"] == "completed"
            assert sections["compute"]["status"] == "completed"
            # Memory should be failed
            assert sections["memory"]["status"] == "failed"
            # Interconnect should still complete despite memory failure
            assert sections["interconnect"]["status"] == "completed"

    def test_persists_on_keyboard_interrupt(self) -> None:
        """VAL-RUNBOOK-009: Report persisted when run is interrupted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            runners = {
                "pre-flight": StubSectionRunner("pre-flight"),
                "compute": StubSectionRunner("compute"),
                "memory": InterruptingSectionRunner("memory"),
                "interconnect": StubSectionRunner("interconnect"),
                "post-flight": StubSectionRunner("post-flight"),
                "manifest": StubSectionRunner("manifest"),
            }
            orch = DurableRunOrchestrator(
                runners=runners,
                output_path=output_path,
            )
            orch.execute()
            assert output_path.exists()
            data = json.loads(output_path.read_text())
            sections = {s["name"]: s for s in data["sections"]}
            # Pre-flight and compute should be completed
            assert sections["pre-flight"]["status"] == "completed"
            assert sections["compute"]["status"] == "completed"

    def test_incremental_persistence(self) -> None:
        """VAL-RUNBOOK-009: Each completed section is persisted incrementally."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            runners = {
                "pre-flight": StubSectionRunner("pre-flight"),
                "compute": StubSectionRunner("compute"),
                "memory": FailingSectionRunner("memory"),
                "interconnect": StubSectionRunner("interconnect"),
                "post-flight": StubSectionRunner("post-flight"),
                "manifest": StubSectionRunner("manifest"),
            }
            orch = DurableRunOrchestrator(
                runners=runners,
                output_path=output_path,
            )
            orch.execute()
            data = json.loads(output_path.read_text())
            # All sections should be present in the report
            section_names = [s["name"] for s in data["sections"]]
            assert "pre-flight" in section_names
            assert "compute" in section_names
            assert "memory" in section_names

    def test_report_has_terminal_status_on_interruption(self) -> None:
        """VAL-RUNBOOK-009: Interrupted sections have terminal status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            runners = {
                "pre-flight": StubSectionRunner("pre-flight"),
                "compute": InterruptingSectionRunner("compute"),
                "memory": StubSectionRunner("memory"),
                "interconnect": StubSectionRunner("interconnect"),
                "post-flight": StubSectionRunner("post-flight"),
                "manifest": StubSectionRunner("manifest"),
            }
            orch = DurableRunOrchestrator(
                runners=runners,
                output_path=output_path,
            )
            orch.execute()
            data = json.loads(output_path.read_text())
            sections = {s["name"]: s for s in data["sections"]}
            # Interrupted section should be marked failed
            assert sections["compute"]["status"] == "failed"
            assert "interrupted" in (sections["compute"].get("error", "")).lower()
