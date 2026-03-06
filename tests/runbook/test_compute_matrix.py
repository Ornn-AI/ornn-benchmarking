"""Tests for compute benchmark matrix (Section 8.1).

Validates VAL-RUNBOOK-002:
- Compute section runs required dtypes (BF16, FP8 E4M3, FP8 E5M2, FP16; TF32 optional)
- Includes fixed-shape checks
- Records per-GPU isolated execution evidence (CUDA_VISIBLE_DEVICES / target GPU id)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ornn_bench.models import BenchmarkStatus
from ornn_bench.runbook.compute import (
    DEFAULT_FIXED_SHAPE,
    REQUIRED_DTYPES,
    ComputeMatrixRunner,
    collect_compute_matrix,
    run_mamf_fixed_shape,
    run_mamf_for_dtype,
)
from ornn_bench.runbook.parsers import parse_mamf_output

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _load_fixture(path: str) -> str:
    return (FIXTURES / path).read_text()


# ---------------------------------------------------------------------------
# mamf-finder parser tests
# ---------------------------------------------------------------------------


class TestMamfParser:
    """Tests for the mamf-finder output parser."""

    def test_parses_dtype(self) -> None:
        raw = _load_fixture("mamf_finder/bf16_range.txt")
        result = parse_mamf_output(raw)
        assert result["dtype"] == "bf16"

    def test_parses_device(self) -> None:
        raw = _load_fixture("mamf_finder/bf16_range.txt")
        result = parse_mamf_output(raw)
        assert "H100" in result["device"]

    def test_parses_results_rows(self) -> None:
        raw = _load_fixture("mamf_finder/bf16_range.txt")
        result = parse_mamf_output(raw)
        assert len(result["results"]) == 4
        for row in result["results"]:
            assert "m" in row
            assert "n" in row
            assert "k" in row
            assert "tflops" in row

    def test_parses_best_result(self) -> None:
        raw = _load_fixture("mamf_finder/bf16_range.txt")
        result = parse_mamf_output(raw)
        assert result["best"] is not None
        assert result["best"]["tflops"] == 891.3

    def test_parses_fp8_e4m3(self) -> None:
        raw = _load_fixture("mamf_finder/fp8_e4m3_range.txt")
        result = parse_mamf_output(raw)
        assert result["dtype"] == "fp8_e4m3"
        assert result["best"]["tflops"] == 1782.6

    def test_parses_fp8_e5m2(self) -> None:
        raw = _load_fixture("mamf_finder/fp8_e5m2_range.txt")
        result = parse_mamf_output(raw)
        assert result["dtype"] == "fp8_e5m2"
        assert result["best"]["tflops"] == 1750.4

    def test_parses_fp16(self) -> None:
        raw = _load_fixture("mamf_finder/fp16_range.txt")
        result = parse_mamf_output(raw)
        assert result["dtype"] == "fp16"
        assert result["best"]["tflops"] == 885.7

    def test_parses_fixed_shape(self) -> None:
        raw = _load_fixture("mamf_finder/fixed_shape.txt")
        result = parse_mamf_output(raw)
        assert result["dtype"] == "bf16"
        assert result["best"] is not None
        assert result["best"]["m"] == 4096
        assert result["best"]["n"] == 4096
        assert result["best"]["k"] == 4096

    def test_empty_input_returns_empty_results(self) -> None:
        result = parse_mamf_output("")
        assert result["results"] == []
        assert result["best"] is None


# ---------------------------------------------------------------------------
# Per-dtype runner tests
# ---------------------------------------------------------------------------


class TestRunMamfForDtype:
    """Tests for running mamf-finder for individual dtypes."""

    def test_bf16_with_fixture(self) -> None:
        raw = _load_fixture("mamf_finder/bf16_range.txt")
        result = run_mamf_for_dtype("bf16", gpu_index=0, raw_output=raw)
        assert result["dtype"] == "bf16"
        assert result["gpu_index"] == 0
        assert result["cuda_visible_devices"] == "0"
        assert result["best"]["tflops"] == 891.3

    def test_fp8_e4m3_with_fixture(self) -> None:
        raw = _load_fixture("mamf_finder/fp8_e4m3_range.txt")
        result = run_mamf_for_dtype("fp8_e4m3", gpu_index=1, raw_output=raw)
        assert result["dtype"] == "fp8_e4m3"
        assert result["gpu_index"] == 1
        assert result["cuda_visible_devices"] == "1"

    def test_records_gpu_isolation(self) -> None:
        """VAL-RUNBOOK-002: Per-GPU isolation evidence."""
        raw = _load_fixture("mamf_finder/bf16_range.txt")
        result = run_mamf_for_dtype("bf16", gpu_index=3, raw_output=raw)
        assert result["cuda_visible_devices"] == "3"
        assert result["gpu_index"] == 3


# ---------------------------------------------------------------------------
# Fixed-shape runner tests
# ---------------------------------------------------------------------------


class TestRunMamfFixedShape:
    """Tests for mamf-finder fixed-shape execution."""

    def test_fixed_shape_with_fixture(self) -> None:
        raw = _load_fixture("mamf_finder/fixed_shape.txt")
        result = run_mamf_fixed_shape("bf16", gpu_index=0, raw_output=raw)
        assert result["fixed_shape"] == DEFAULT_FIXED_SHAPE
        assert result["best"]["tflops"] == 789.2

    def test_fixed_shape_records_gpu_index(self) -> None:
        raw = _load_fixture("mamf_finder/fixed_shape.txt")
        result = run_mamf_fixed_shape("bf16", gpu_index=2, raw_output=raw)
        assert result["gpu_index"] == 2
        assert result["cuda_visible_devices"] == "2"

    def test_custom_shape(self) -> None:
        raw = _load_fixture("mamf_finder/fixed_shape.txt")
        shape = {"m": 2048, "n": 2048, "k": 2048}
        result = run_mamf_fixed_shape("bf16", gpu_index=0, shape=shape, raw_output=raw)
        assert result["fixed_shape"] == shape


# ---------------------------------------------------------------------------
# Full compute matrix tests
# ---------------------------------------------------------------------------


class TestCollectComputeMatrix:
    """Tests for the full compute benchmark matrix collection."""

    def _dtype_outputs(self) -> dict[str, str]:
        return {
            "bf16": _load_fixture("mamf_finder/bf16_range.txt"),
            "fp8_e4m3": _load_fixture("mamf_finder/fp8_e4m3_range.txt"),
            "fp8_e5m2": _load_fixture("mamf_finder/fp8_e5m2_range.txt"),
            "fp16": _load_fixture("mamf_finder/fp16_range.txt"),
        }

    def test_covers_required_dtypes(self) -> None:
        """VAL-RUNBOOK-002: All required dtypes covered."""
        result = collect_compute_matrix(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        assert set(REQUIRED_DTYPES).issubset(set(result["dtypes_tested"]))

    def test_per_gpu_results_present(self) -> None:
        """VAL-RUNBOOK-002: Per-GPU results present."""
        result = collect_compute_matrix(
            gpu_count=2,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        assert "gpu_0" in result["per_gpu"]
        assert "gpu_1" in result["per_gpu"]

    def test_detected_gpu_count_runs_all_detected_gpus(self) -> None:
        """VAL-CORE-001: Detection path fans out compute across all GPUs."""
        with patch("ornn_bench.runbook.compute.detect_gpu_count", return_value=8):
            result = collect_compute_matrix(
                dtype_outputs=self._dtype_outputs(),
                fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
            )

        assert result["gpu_count"] == 8
        assert set(result["per_gpu"]) == {f"gpu_{index}" for index in range(8)}
        for index in range(8):
            gpu_key = f"gpu_{index}"
            assert result["fixed_shape_results"][gpu_key]["cuda_visible_devices"] == str(index)
            for dtype in REQUIRED_DTYPES:
                assert result["per_gpu"][gpu_key][dtype]["cuda_visible_devices"] == str(index)

    def test_each_gpu_has_all_dtypes(self) -> None:
        """VAL-RUNBOOK-002: Each GPU has results for all dtypes."""
        result = collect_compute_matrix(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        gpu_results = result["per_gpu"]["gpu_0"]
        for dtype in REQUIRED_DTYPES:
            assert dtype in gpu_results
            assert gpu_results[dtype]["best"] is not None

    def test_includes_fixed_shape_results(self) -> None:
        """VAL-RUNBOOK-002: Fixed-shape checks included."""
        result = collect_compute_matrix(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        assert "gpu_0" in result["fixed_shape_results"]
        fs = result["fixed_shape_results"]["gpu_0"]
        assert fs["fixed_shape"] == DEFAULT_FIXED_SHAPE

    def test_per_gpu_isolation_evidence(self) -> None:
        """VAL-RUNBOOK-002: Per-GPU isolation evidence present."""
        result = collect_compute_matrix(
            gpu_count=2,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        for gpu_key in ("gpu_0", "gpu_1"):
            for dtype in REQUIRED_DTYPES:
                assert "cuda_visible_devices" in result["per_gpu"][gpu_key][dtype]

    def test_tf32_optional_excluded_by_default(self) -> None:
        """TF32 not included unless explicitly enabled."""
        result = collect_compute_matrix(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        assert "tf32" not in result["dtypes_tested"]

    def test_tf32_included_when_enabled(self) -> None:
        """TF32 included when include_tf32=True."""
        result = collect_compute_matrix(
            gpu_count=1,
            include_tf32=True,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        assert "tf32" in result["dtypes_tested"]


# ---------------------------------------------------------------------------
# ComputeMatrixRunner section runner tests
# ---------------------------------------------------------------------------


class TestComputeMatrixRunner:
    """Tests for the ComputeMatrixRunner section runner."""

    def _dtype_outputs(self) -> dict[str, str]:
        return {
            "bf16": _load_fixture("mamf_finder/bf16_range.txt"),
            "fp8_e4m3": _load_fixture("mamf_finder/fp8_e4m3_range.txt"),
            "fp8_e5m2": _load_fixture("mamf_finder/fp8_e5m2_range.txt"),
            "fp16": _load_fixture("mamf_finder/fp16_range.txt"),
        }

    def test_runner_returns_completed(self) -> None:
        runner = ComputeMatrixRunner(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        result = runner.run()
        assert result.status == BenchmarkStatus.COMPLETED

    def test_runner_has_timestamps(self) -> None:
        runner = ComputeMatrixRunner(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        result = runner.run()
        assert result.started_at is not None
        assert result.finished_at is not None

    def test_runner_name_is_compute(self) -> None:
        runner = ComputeMatrixRunner(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        result = runner.run()
        assert result.name == "compute"

    def test_runner_metrics_have_dtypes(self) -> None:
        runner = ComputeMatrixRunner(
            gpu_count=1,
            dtype_outputs=self._dtype_outputs(),
            fixed_shape_outputs={"bf16": _load_fixture("mamf_finder/fixed_shape.txt")},
        )
        result = runner.run()
        assert "dtypes_tested" in result.metrics
        assert "per_gpu" in result.metrics
