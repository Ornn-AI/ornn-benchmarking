"""Smoke tests to verify project structure and basic imports."""

from __future__ import annotations


def test_version_importable() -> None:
    """Verify the package version is importable."""
    from ornn_bench import __version__

    assert __version__ == "0.2.0"


def test_cli_app_importable() -> None:
    """Verify the CLI app object is importable."""
    from ornn_bench.cli import app

    assert app is not None


def test_models_importable() -> None:
    """Verify core data models are importable and constructable."""
    from ornn_bench.models import (
        BenchmarkReport,
        BenchmarkStatus,
        GPUInfo,
        Qualification,
        ScoreResult,
        SectionResult,
        SystemInventory,
    )

    gpu = GPUInfo(uuid="test-uuid", name="Test GPU")
    assert gpu.uuid == "test-uuid"

    inventory = SystemInventory(gpus=[gpu])
    assert len(inventory.gpus) == 1

    section = SectionResult(name="compute", status=BenchmarkStatus.PENDING)
    assert section.status == BenchmarkStatus.PENDING

    report = BenchmarkReport(
        report_id="smoke-test",
        sections=[section],
    )
    assert report.schema_version == "1.0.0"
    assert len(report.sections) == 1

    score = ScoreResult(ornn_i=85.0, qualification=Qualification.PREMIUM)
    assert score.ornn_i == 85.0


def test_scoring_importable() -> None:
    """Verify the scoring module is importable."""
    from ornn_bench.scoring import compute_ornn_i, compute_ornn_t, compute_scores

    assert callable(compute_ornn_i)
    assert callable(compute_ornn_t)
    assert callable(compute_scores)
