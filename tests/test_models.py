"""Tests for Pydantic data models."""

from __future__ import annotations

from ornn_bench.models import (
    BenchmarkReport,
    BenchmarkStatus,
    GPUInfo,
    Qualification,
    ScoreResult,
    SectionResult,
    SystemInventory,
)


class TestGPUInfo:
    """Tests for GPUInfo model."""

    def test_defaults(self) -> None:
        """GPUInfo has sensible defaults."""
        gpu = GPUInfo()
        assert gpu.uuid == ""
        assert gpu.name == ""
        assert gpu.memory_total_mb == 0

    def test_roundtrip_json(self) -> None:
        """GPUInfo serializes and deserializes correctly."""
        gpu = GPUInfo(uuid="GPU-123", name="H100", memory_total_mb=81920)
        data = gpu.model_dump()
        restored = GPUInfo.model_validate(data)
        assert restored.uuid == gpu.uuid
        assert restored.name == gpu.name


class TestSystemInventory:
    """Tests for SystemInventory model."""

    def test_empty_defaults(self) -> None:
        """SystemInventory works with all defaults."""
        inv = SystemInventory()
        assert inv.gpus == []
        assert inv.os_info == ""

    def test_with_gpus(self, sample_gpu_info: GPUInfo) -> None:
        """SystemInventory holds GPU list."""
        inv = SystemInventory(gpus=[sample_gpu_info])
        assert len(inv.gpus) == 1
        assert inv.gpus[0].name == "NVIDIA H100 80GB HBM3"


class TestSectionResult:
    """Tests for SectionResult model."""

    def test_defaults(self) -> None:
        """SectionResult defaults to PENDING with no error."""
        section = SectionResult(name="compute")
        assert section.status == BenchmarkStatus.PENDING
        assert section.error is None
        assert section.metrics == {}

    def test_failed_section(self) -> None:
        """Failed section carries error message."""
        section = SectionResult(
            name="memory",
            status=BenchmarkStatus.FAILED,
            error="nvbandwidth not found",
        )
        assert section.status == BenchmarkStatus.FAILED
        assert "nvbandwidth" in (section.error or "")


class TestScoreResult:
    """Tests for ScoreResult model."""

    def test_defaults_none(self) -> None:
        """ScoreResult defaults to None scores."""
        score = ScoreResult()
        assert score.ornn_i is None
        assert score.ornn_t is None
        assert score.qualification is None

    def test_qualification_values(self) -> None:
        """Qualification enum values are correct strings."""
        assert Qualification.PREMIUM == "Premium"
        assert Qualification.STANDARD == "Standard"
        assert Qualification.BELOW == "Below"


class TestBenchmarkReport:
    """Tests for BenchmarkReport model."""

    def test_defaults(self) -> None:
        """BenchmarkReport has proper defaults."""
        report = BenchmarkReport()
        assert report.schema_version == "1.0.0"
        assert report.sections == []
        assert report.scores.ornn_i is None

    def test_json_roundtrip(self, sample_report: BenchmarkReport) -> None:
        """Report survives JSON serialization roundtrip."""
        json_str = sample_report.model_dump_json()
        restored = BenchmarkReport.model_validate_json(json_str)
        assert restored.report_id == sample_report.report_id
        assert len(restored.sections) == len(sample_report.sections)
        assert restored.scores.ornn_i == sample_report.scores.ornn_i

    def test_section_status_values(self) -> None:
        """BenchmarkStatus enum covers expected states."""
        expected = {"pending", "running", "completed", "failed", "skipped", "timeout"}
        actual = {s.value for s in BenchmarkStatus}
        assert actual == expected
