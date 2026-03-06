"""Integration-style tests for deriving report scores from runbook sections.

Covers normalized ratio extraction, per-GPU minimum aggregation,
and compatibility with server-side verification using refs=1.0.
"""

from __future__ import annotations

import pytest
from api.scoring import VerificationStatus, recompute_and_verify

from ornn_bench.models import (
    BenchmarkStatus,
    Qualification,
    ScoreStatus,
    SectionResult,
)
from ornn_bench.runner import SECTION_ORDER, RunOrchestrator, SectionRunner


class StubSectionRunner(SectionRunner):
    """A section runner that succeeds with empty metrics."""

    def run(self) -> SectionResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            metrics={},
        )


class MetricsSectionRunner(SectionRunner):
    """A section runner that succeeds with predefined metrics."""

    def __init__(self, name: str, metrics: dict[str, object]) -> None:
        super().__init__(name)
        self._metrics = metrics

    def run(self) -> SectionResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.COMPLETED,
            started_at=now,
            finished_at=now,
            metrics=self._metrics,
        )


class FailedSectionRunner(SectionRunner):
    """A section runner that fails with a configured error message."""

    def __init__(self, name: str, error: str) -> None:
        super().__init__(name)
        self._error = error

    def run(self) -> SectionResult:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        return SectionResult(
            name=self.name,
            status=BenchmarkStatus.FAILED,
            started_at=now,
            finished_at=now,
            error=self._error,
        )


def _preflight_metrics() -> dict[str, object]:
    return {
        "os": "Ubuntu 22.04.3 LTS",
        "kernel": "5.15.0-91-generic",
        "cpu_model": "Intel(R) Xeon(R) Platinum 8480+",
        "numa_nodes": 2,
        "pytorch_version": "2.1.2",
        "driver_version": "535.129.03",
        "cuda_version": "12.2",
        "software_versions": {
            "driver": "535.129.03",
            "cuda": "12.2",
            "python": "3.10.13",
        },
        "gpu_inventory": {
            "gpus": [
                {
                    "uuid": "GPU-AAA",
                    "product_name": "NVIDIA H100 80GB HBM3",
                    "memory_total_mib": 81920,
                },
                {
                    "uuid": "GPU-BBB",
                    "product_name": "NVIDIA H100 80GB HBM3",
                    "memory_total_mib": 81920,
                },
            ]
        },
    }


def _compute_metrics(
    *,
    gpu0_fp8_e4m3: float = 1782.6,
    gpu0_fp8_e5m2: float = 1750.4,
    gpu0_bf16: float = 891.3,
    gpu1_fp8_e4m3: float = 1782.6,
    gpu1_fp8_e5m2: float = 1750.4,
    gpu1_bf16: float = 891.3,
) -> dict[str, object]:
    return {
        "gpu_count": 2,
        "per_gpu": {
            "gpu_0": {
                "bf16": {"best": {"tflops": gpu0_bf16}},
                "fp8_e4m3": {"best": {"tflops": gpu0_fp8_e4m3}},
                "fp8_e5m2": {"best": {"tflops": gpu0_fp8_e5m2}},
            },
            "gpu_1": {
                "bf16": {"best": {"tflops": gpu1_bf16}},
                "fp8_e4m3": {"best": {"tflops": gpu1_fp8_e4m3}},
                "fp8_e5m2": {"best": {"tflops": gpu1_fp8_e5m2}},
            },
        },
        "fixed_shape_results": {},
    }


def _memory_metrics(*, device_local_copy_max: float = 2039.5) -> dict[str, object]:
    return {
        "nvbandwidth_results": {
            "device_local_copy": {"max": device_local_copy_max},
            "device_local_copy_sm": {"max": 9999.0},
            "d2d_bidir": {"max": 4079.0},
        }
    }


def _interconnect_metrics(*, all_reduce_1gb_avg: float = 148.32) -> dict[str, object]:
    return {
        "gpu_count": 2,
        "nccl_results": {
            "all_reduce_1gb": {"avg_bus_bandwidth": all_reduce_1gb_avg},
            "all_reduce_sweep": {"avg_bus_bandwidth": 999.0},
        },
        "bus_bandwidth_summary": {
            "all_reduce_1gb": {"avg_busbw": all_reduce_1gb_avg, "max_busbw": all_reduce_1gb_avg},
            "all_reduce_sweep": {"avg_busbw": 999.0, "max_busbw": 999.0},
        },
    }


def _make_scoring_runners(
    *,
    memory_runner: SectionRunner | None = None,
    compute_metrics: dict[str, object] | None = None,
) -> dict[str, SectionRunner]:
    runners: dict[str, SectionRunner] = {
        name: StubSectionRunner(name) for name in SECTION_ORDER
    }
    runners["pre-flight"] = MetricsSectionRunner("pre-flight", _preflight_metrics())
    runners["compute"] = MetricsSectionRunner(
        "compute", compute_metrics or _compute_metrics()
    )
    runners["memory"] = memory_runner or MetricsSectionRunner(
        "memory", _memory_metrics()
    )
    runners["interconnect"] = MetricsSectionRunner(
        "interconnect", _interconnect_metrics()
    )
    return runners


class TestReportScoringFromRunbookSections:
    """Tests for score derivation from completed runbook section metrics."""

    def test_report_scores_use_normalized_components_from_documented_metrics(self) -> None:
        """BW/FP8/BF16/AR ratios are derived from the documented source metrics."""
        report = RunOrchestrator(runners=_make_scoring_runners()).execute()

        assert report.scores.score_status == ScoreStatus.VALID
        assert report.scores.qualification == Qualification.PREMIUM
        assert report.scores.aggregate_method == "minimum"
        assert report.scores.components == pytest.approx(
            {"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0}
        )
        assert report.scores.ornn_i == pytest.approx(100.0)
        assert report.scores.ornn_t == pytest.approx(100.0)
        assert len(report.scores.per_gpu_scores) == 2
        assert [record.gpu_uuid for record in report.scores.per_gpu_scores] == [
            "GPU-AAA",
            "GPU-BBB",
        ]

    def test_server_verification_matches_when_gpu_minima_differ_by_score_type(self) -> None:
        """Aggregate components allow server recomputation to match local min-GPU scores."""
        compute_metrics = _compute_metrics(
            gpu0_fp8_e4m3=1600.0,
            gpu0_fp8_e5m2=1500.0,
            gpu0_bf16=891.3,
            gpu1_fp8_e4m3=1782.6,
            gpu1_fp8_e5m2=1750.4,
            gpu1_bf16=800.0,
        )
        report = RunOrchestrator(
            runners=_make_scoring_runners(compute_metrics=compute_metrics)
        ).execute()

        expected_fp8_ratio = 1600.0 / 1782.6
        expected_bf16_ratio = 800.0 / 891.3
        expected_ornn_i = 55.0 * 1.0 + 45.0 * expected_fp8_ratio
        expected_ornn_t = 55.0 * expected_bf16_ratio + 45.0 * 1.0

        assert report.scores.score_status == ScoreStatus.VALID
        assert report.scores.aggregate_method == "minimum"
        assert report.scores.components == pytest.approx(
            {
                "bw": 1.0,
                "fp8": expected_fp8_ratio,
                "bf16": expected_bf16_ratio,
                "ar": 1.0,
            }
        )
        assert report.scores.ornn_i == pytest.approx(expected_ornn_i)
        assert report.scores.ornn_t == pytest.approx(expected_ornn_t)

        qualification = (
            report.scores.qualification.value if report.scores.qualification else None
        )
        verification = recompute_and_verify(
            components=report.scores.components,
            submitted_ornn_i=report.scores.ornn_i,
            submitted_ornn_t=report.scores.ornn_t,
            submitted_qualification=qualification,
        )
        assert verification.status == VerificationStatus.VERIFIED

    def test_failed_memory_section_produces_partial_score_with_helpful_detail(self) -> None:
        """Missing BW keeps training score valid and inference score partial with context."""
        memory_runner = FailedSectionRunner("memory", "nvbandwidth missing")
        report = RunOrchestrator(
            runners=_make_scoring_runners(memory_runner=memory_runner)
        ).execute()

        assert report.scores.score_status == ScoreStatus.PARTIAL
        assert report.scores.ornn_i is None
        assert report.scores.ornn_t == pytest.approx(100.0)
        assert report.scores.aggregate_method == "minimum"
        assert len(report.scores.per_gpu_scores) == 2
        assert report.scores.components == pytest.approx(
            {"fp8": 1.0, "bf16": 1.0, "ar": 1.0}
        )
        assert report.scores.score_status_detail is not None
        assert "bw" in report.scores.score_status_detail.lower()
        assert "memory" in report.scores.score_status_detail.lower()
