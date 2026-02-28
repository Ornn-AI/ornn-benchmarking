"""Tests for invalid/missing metric handling in scoring (VAL-RUNBOOK-011).

Ensures that missing or invalid component metrics produce explicit error/skip
states rather than silent NaN success.
"""

from __future__ import annotations

import math

from ornn_bench.models import ScoreStatus
from ornn_bench.scoring import (
    PerGPUScore,
    aggregate_gpu_scores,
    compute_ornn_i,
    compute_ornn_t,
    compute_scores,
)


class TestMissingMetricValues:
    """Verify behavior when required metric values are None."""

    def test_ornn_i_none_bw(self) -> None:
        """Missing BW → Ornn-I is None."""
        assert compute_ornn_i(None, 1.0) is None

    def test_ornn_i_none_fp8(self) -> None:
        """Missing FP8 → Ornn-I is None."""
        assert compute_ornn_i(1.0, None) is None

    def test_ornn_i_both_none(self) -> None:
        """Both missing → Ornn-I is None."""
        assert compute_ornn_i(None, None) is None

    def test_ornn_t_none_bf16(self) -> None:
        """Missing BF16 → Ornn-T is None."""
        assert compute_ornn_t(None, 1.0) is None

    def test_ornn_t_none_ar(self) -> None:
        """Missing AR → Ornn-T is None."""
        assert compute_ornn_t(1.0, None) is None

    def test_ornn_t_both_none(self) -> None:
        """Both missing → Ornn-T is None."""
        assert compute_ornn_t(None, None) is None


class TestInvalidReferenceValues:
    """Verify behavior when reference values are invalid (zero/negative)."""

    def test_zero_bw_ref(self) -> None:
        """Zero BW reference → Ornn-I is None (no division by zero)."""
        assert compute_ornn_i(1.0, 1.0, bw_ref=0.0) is None

    def test_zero_fp8_ref(self) -> None:
        """Zero FP8 reference → Ornn-I is None."""
        assert compute_ornn_i(1.0, 1.0, fp8_ref=0.0) is None

    def test_negative_bw_ref(self) -> None:
        """Negative BW reference → Ornn-I is None."""
        assert compute_ornn_i(1.0, 1.0, bw_ref=-1.0) is None

    def test_negative_fp8_ref(self) -> None:
        """Negative FP8 reference → Ornn-I is None."""
        assert compute_ornn_i(1.0, 1.0, fp8_ref=-1.0) is None

    def test_zero_bf16_ref(self) -> None:
        """Zero BF16 reference → Ornn-T is None."""
        assert compute_ornn_t(1.0, 1.0, bf16_ref=0.0) is None

    def test_zero_ar_ref(self) -> None:
        """Zero AR reference → Ornn-T is None."""
        assert compute_ornn_t(1.0, 1.0, ar_ref=0.0) is None

    def test_negative_bf16_ref(self) -> None:
        """Negative BF16 reference → Ornn-T is None."""
        assert compute_ornn_t(1.0, 1.0, bf16_ref=-1.0) is None


class TestNaNAndInfInputs:
    """Verify NaN and Inf metric values are rejected (VAL-RUNBOOK-011)."""

    def test_nan_bw_produces_none(self) -> None:
        """NaN metric value → Ornn-I is None, not silent NaN success."""
        result = compute_ornn_i(float("nan"), 1.0)
        assert result is None

    def test_nan_fp8_produces_none(self) -> None:
        """NaN FP8 → Ornn-I is None."""
        result = compute_ornn_i(1.0, float("nan"))
        assert result is None

    def test_inf_bw_produces_none(self) -> None:
        """Inf metric value → Ornn-I is None."""
        result = compute_ornn_i(float("inf"), 1.0)
        assert result is None

    def test_neg_inf_produces_none(self) -> None:
        """Negative Inf → Ornn-I is None."""
        result = compute_ornn_i(float("-inf"), 1.0)
        assert result is None

    def test_nan_bf16_produces_none(self) -> None:
        """NaN BF16 → Ornn-T is None."""
        result = compute_ornn_t(float("nan"), 1.0)
        assert result is None

    def test_inf_ar_produces_none(self) -> None:
        """Inf AR → Ornn-T is None."""
        result = compute_ornn_t(1.0, float("inf"))
        assert result is None


class TestNegativeMetricValues:
    """Verify that negative metric values are rejected."""

    def test_negative_bw_produces_none(self) -> None:
        """Negative BW → Ornn-I is None."""
        result = compute_ornn_i(-1.0, 1.0)
        assert result is None

    def test_negative_fp8_produces_none(self) -> None:
        """Negative FP8 → Ornn-I is None."""
        result = compute_ornn_i(1.0, -1.0)
        assert result is None

    def test_negative_bf16_produces_none(self) -> None:
        """Negative BF16 → Ornn-T is None."""
        result = compute_ornn_t(-1.0, 1.0)
        assert result is None

    def test_negative_ar_produces_none(self) -> None:
        """Negative AR → Ornn-T is None."""
        result = compute_ornn_t(1.0, -1.0)
        assert result is None


class TestScoreStatusExplicitness:
    """Verify ScoreStatus is explicit and never silently successful (VAL-RUNBOOK-011)."""

    def test_all_valid_inputs_status_valid(self) -> None:
        """All metrics provided → VALID status."""
        result = compute_scores(bw=1.0, fp8=1.0, bf16=1.0, ar=1.0)
        assert result.score_status == ScoreStatus.VALID

    def test_only_inference_metrics_status_partial(self) -> None:
        """Only Ornn-I metrics → PARTIAL status."""
        result = compute_scores(bw=1.0, fp8=1.0)
        assert result.score_status == ScoreStatus.PARTIAL
        assert result.score_status_detail is not None
        assert "training" in result.score_status_detail.lower() or "ornn_t" in result.score_status_detail.lower()

    def test_only_training_metrics_status_partial(self) -> None:
        """Only Ornn-T metrics → PARTIAL status."""
        result = compute_scores(bf16=1.0, ar=1.0)
        assert result.score_status == ScoreStatus.PARTIAL

    def test_no_metrics_status_error(self) -> None:
        """No metrics → ERROR status."""
        result = compute_scores()
        assert result.score_status == ScoreStatus.ERROR
        assert result.score_status_detail is not None

    def test_nan_input_status_error(self) -> None:
        """NaN inputs → score is None, status reflects error."""
        result = compute_scores(bw=float("nan"), fp8=1.0, bf16=1.0, ar=1.0)
        assert result.ornn_i is None
        assert result.score_status in (ScoreStatus.PARTIAL, ScoreStatus.ERROR)

    def test_score_never_nan(self) -> None:
        """Scores must never be NaN (VAL-RUNBOOK-011 core invariant)."""
        test_cases = [
            {"bw": float("nan"), "fp8": 1.0},
            {"bw": 1.0, "fp8": float("nan")},
            {"bf16": float("nan"), "ar": 1.0},
            {"bw": float("inf"), "fp8": 1.0},
        ]
        for kwargs in test_cases:
            result = compute_scores(**kwargs)  # type: ignore[arg-type]
            if result.ornn_i is not None:
                assert not math.isnan(result.ornn_i), f"Ornn-I is NaN for {kwargs}"
                assert not math.isinf(result.ornn_i), f"Ornn-I is Inf for {kwargs}"
            if result.ornn_t is not None:
                assert not math.isnan(result.ornn_t), f"Ornn-T is NaN for {kwargs}"
                assert not math.isinf(result.ornn_t), f"Ornn-T is Inf for {kwargs}"


class TestAggregationWithInvalidGPUs:
    """Test multi-GPU aggregation with invalid per-GPU metric data."""

    def test_all_gpus_missing_all_metrics(self) -> None:
        """All GPUs have no metrics → ERROR status."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001"),
            PerGPUScore(gpu_uuid="GPU-0002"),
        ]
        result = aggregate_gpu_scores(per_gpu)
        assert result.ornn_i is None
        assert result.ornn_t is None
        assert result.score_status == ScoreStatus.ERROR

    def test_one_gpu_valid_one_invalid(self) -> None:
        """One valid GPU, one with no metrics → partial aggregate."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
            PerGPUScore(gpu_uuid="GPU-0002"),
        ]
        result = aggregate_gpu_scores(per_gpu)
        # Second GPU has no scores → overall should reflect partial
        assert result.score_status in (ScoreStatus.PARTIAL, ScoreStatus.ERROR)

    def test_nan_metric_in_gpu(self) -> None:
        """GPU with NaN metric → that GPU score is None, handled gracefully."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
            PerGPUScore(gpu_uuid="GPU-0002", bw=float("nan"), fp8=1.0, bf16=1.0, ar=1.0),
        ]
        result = aggregate_gpu_scores(per_gpu)
        # GPU-0002's ornn_i should be None due to NaN
        assert result.score_status in (ScoreStatus.PARTIAL, ScoreStatus.ERROR)
        # Must not produce NaN aggregate
        if result.ornn_i is not None:
            assert not math.isnan(result.ornn_i)

    def test_aggregate_never_nan(self) -> None:
        """Aggregate scores must never be NaN (core safety invariant)."""
        per_gpu = [
            PerGPUScore(gpu_uuid="G1", bw=float("nan"), fp8=float("nan")),
            PerGPUScore(gpu_uuid="G2", bw=float("inf"), fp8=float("inf")),
        ]
        result = aggregate_gpu_scores(per_gpu)
        if result.ornn_i is not None:
            assert not math.isnan(result.ornn_i)
        if result.ornn_t is not None:
            assert not math.isnan(result.ornn_t)
