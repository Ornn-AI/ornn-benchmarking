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
    SECTION_ORDER,
    DurableRunOrchestrator,
    RunOrchestrator,
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


class MetricsSectionRunner(SectionRunner):
    """Runner that returns completed results with predefined metrics."""

    def __init__(self, name: str, metrics: dict[str, object]) -> None:
        super().__init__(name)
        self._metrics = metrics

    def run(self) -> SectionResult:
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.COMPLETED,
            started_at="2024-01-01T00:00:00Z",
            finished_at="2024-01-01T00:00:01Z",
            metrics=self._metrics,
        )


def _sample_preflight_metrics() -> dict[str, object]:
    return {
        "os": "Ubuntu 22.04.3 LTS",
        "os_version": "#101-Ubuntu SMP Fri Nov 10 00:00:00 UTC 2024",
        "kernel": "5.15.0-91-generic",
        "cpu_model": "Intel(R) Xeon(R) Platinum 8480+",
        "numa_nodes": 2,
        "pytorch_version": "2.1.2",
        "driver_version": "535.129.03",
        "cuda_version": "12.2",
        "software_versions": {
            "python": "3.10.13",
            "driver": "535.129.03",
            "cuda": "12.2",
        },
        "gpu_inventory": {
            "attached_gpus": 2,
            "gpu_uuids": [
                "GPU-12345678-abcd-1234-abcd-123456789abc",
                "GPU-87654321-dcba-4321-dcba-cba987654321",
            ],
            "gpus": [
                {
                    "uuid": "GPU-12345678-abcd-1234-abcd-123456789abc",
                    "product_name": "NVIDIA H100 80GB HBM3",
                    "memory_total_mib": 81920,
                },
                {
                    "uuid": "GPU-87654321-dcba-4321-dcba-cba987654321",
                    "product_name": "NVIDIA H100 80GB HBM3",
                    "memory_total_mib": 81920,
                },
            ],
        },
        "nvlink_topology": [],
        "ecc_baseline": {},
    }


def _make_enriched_runners() -> dict[str, SectionRunner]:
    runners = {name: StubSectionRunner(name) for name in SECTION_ORDER}
    runners["pre-flight"] = MetricsSectionRunner("pre-flight", _sample_preflight_metrics())
    runners["manifest"] = ManifestRunner()
    return runners


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


class TestBenchmarkReportEnrichment:
    """Tests for top-level report enrichment from section outputs."""

    def test_orchestrator_maps_preflight_metrics_to_system_inventory(self) -> None:
        """VAL-CORE-004: system_inventory is derived from pre-flight metrics."""
        report = RunOrchestrator(runners=_make_enriched_runners()).execute()

        assert report.system_inventory.os_info == "Ubuntu 22.04.3 LTS"
        assert report.system_inventory.kernel_version == "5.15.0-91-generic"
        assert report.system_inventory.cpu_model == "Intel(R) Xeon(R) Platinum 8480+"
        assert report.system_inventory.numa_nodes == 2
        assert report.system_inventory.pytorch_version == "2.1.2"
        assert len(report.system_inventory.gpus) == 2

        first_gpu = report.system_inventory.gpus[0]
        assert first_gpu.uuid == "GPU-12345678-abcd-1234-abcd-123456789abc"
        assert first_gpu.name == "NVIDIA H100 80GB HBM3"
        assert first_gpu.driver_version == "535.129.03"
        assert first_gpu.cuda_version == "12.2"
        assert first_gpu.memory_total_mb == 81920

    def test_orchestrator_maps_manifest_metrics_to_report_manifest(self) -> None:
        """VAL-CORE-005: manifest section metrics populate report.manifest."""
        report = RunOrchestrator(runners=_make_enriched_runners()).execute()

        manifest_section = next(
            section for section in report.sections if section.name == "manifest"
        )
        assert report.manifest == manifest_section.metrics
        assert report.manifest["entries"]
        summary = report.manifest["summary"]
        assert isinstance(summary, dict)
        assert summary["produced"] > 0
        assert summary["total"] >= summary["produced"]


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

    def test_persisted_report_includes_system_inventory_and_manifest(self) -> None:
        """VAL-CORE-004/005: persisted JSON includes enriched top-level fields."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"
            orch = DurableRunOrchestrator(
                runners=_make_enriched_runners(),
                output_path=output_path,
            )

            orch.execute()

            data = json.loads(output_path.read_text())
            assert data["system_inventory"]["kernel_version"] == "5.15.0-91-generic"
            assert data["system_inventory"]["gpus"][0]["uuid"] == (
                "GPU-12345678-abcd-1234-abcd-123456789abc"
            )
            assert data["manifest"]["entries"]
            assert data["manifest"]["summary"]["produced"] > 0
