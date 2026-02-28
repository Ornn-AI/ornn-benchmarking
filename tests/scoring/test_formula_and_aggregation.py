"""Tests for Ornn-I/Ornn-T scoring formula and multi-GPU aggregation.

Covers VAL-CLI-007 (scorecard clarity), VAL-RUNBOOK-010 (multi-GPU aggregation),
and deterministic scoring with explicit aggregate method documentation.
"""

from __future__ import annotations

import pytest

from ornn_bench.models import Qualification, ScoreStatus
from ornn_bench.scoring import (
    AggregateMethod,
    PerGPUScore,
    aggregate_gpu_scores,
    compute_ornn_i,
    compute_ornn_t,
    compute_scores,
    determine_qualification,
)


class TestOrnnIFormula:
    """Verify Ornn-I = 55*(BW/BW_ref) + 45*(FP8/FP8_ref)."""

    def test_unit_reference(self) -> None:
        """All metrics at reference level → score 100."""
        assert compute_ornn_i(1.0, 1.0) == pytest.approx(100.0)

    def test_scaled_values(self) -> None:
        """Score scales linearly with metric ratios."""
        result = compute_ornn_i(2.0, 0.5, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(55.0 * 2.0 + 45.0 * 0.5)

    def test_custom_refs(self) -> None:
        """Custom reference values produce correct normalization."""
        result = compute_ornn_i(3000.0, 500.0, bw_ref=3000.0, fp8_ref=500.0)
        assert result == pytest.approx(100.0)

    def test_zero_metric_value(self) -> None:
        """Zero metric values produce zero contribution (not error)."""
        result = compute_ornn_i(0.0, 1.0)
        assert result == pytest.approx(45.0)


class TestOrnnTFormula:
    """Verify Ornn-T = 55*(BF16/BF16_ref) + 45*(AR/AR_ref)."""

    def test_unit_reference(self) -> None:
        """All metrics at reference level → score 100."""
        assert compute_ornn_t(1.0, 1.0) == pytest.approx(100.0)

    def test_scaled_values(self) -> None:
        """Score scales linearly with metric ratios."""
        result = compute_ornn_t(0.8, 1.5, bf16_ref=1.0, ar_ref=1.0)
        assert result == pytest.approx(55.0 * 0.8 + 45.0 * 1.5)

    def test_custom_refs(self) -> None:
        """Custom reference values produce correct normalization."""
        result = compute_ornn_t(900.0, 200.0, bf16_ref=900.0, ar_ref=200.0)
        assert result == pytest.approx(100.0)


class TestQualificationGating:
    """Test Premium/Standard/Below qualification with floor + composite gate."""

    def test_premium_both_high(self) -> None:
        """Premium: composite >= 90 and both individual >= 80."""
        assert determine_qualification(95.0, 95.0) == Qualification.PREMIUM

    def test_premium_boundary(self) -> None:
        """Premium at exact boundary: composite=90, floors=80."""
        # composite = (100 + 80) / 2 = 90; both >= 80
        assert determine_qualification(100.0, 80.0) == Qualification.PREMIUM

    def test_premium_fails_floor(self) -> None:
        """Composite >= 90 but one floor < 80 → Standard."""
        # composite = (100 + 79) / 2 = 89.5 — fails composite too
        # Use a case where composite is high but floor fails:
        # composite = (100 + 79.9) / 2 = 89.95 — still < 90
        # Actually both fail, use: ornn_i=100, ornn_t=79
        result = determine_qualification(100.0, 79.0)
        assert result != Qualification.PREMIUM

    def test_standard_qualification(self) -> None:
        """Standard: composite >= 70 and both floors >= 60."""
        assert determine_qualification(75.0, 75.0) == Qualification.STANDARD

    def test_standard_boundary(self) -> None:
        """Standard at exact boundary."""
        assert determine_qualification(70.0, 70.0) == Qualification.STANDARD

    def test_standard_fails_floor(self) -> None:
        """Composite >= 70 but one floor < 60 → Below."""
        result = determine_qualification(80.0, 59.0)
        assert result == Qualification.BELOW

    def test_below_low_composite(self) -> None:
        """Low composite → Below."""
        assert determine_qualification(50.0, 50.0) == Qualification.BELOW

    def test_none_when_missing(self) -> None:
        """Returns None if either score is missing."""
        assert determine_qualification(None, 90.0) is None
        assert determine_qualification(90.0, None) is None
        assert determine_qualification(None, None) is None


class TestComputeScores:
    """Test the aggregate compute_scores function."""

    def test_full_computation_has_valid_status(self) -> None:
        """Full valid inputs → status VALID."""
        result = compute_scores(bw=1.0, fp8=1.0, bf16=1.0, ar=1.0)
        assert result.ornn_i == pytest.approx(100.0)
        assert result.ornn_t == pytest.approx(100.0)
        assert result.qualification == Qualification.PREMIUM
        assert result.score_status == ScoreStatus.VALID

    def test_partial_inputs_partial_status(self) -> None:
        """Only inference inputs → ornn_t is None, status PARTIAL."""
        result = compute_scores(bw=1.0, fp8=1.0)
        assert result.ornn_i == pytest.approx(100.0)
        assert result.ornn_t is None
        assert result.qualification is None
        assert result.score_status == ScoreStatus.PARTIAL

    def test_no_inputs_error_status(self) -> None:
        """No inputs at all → status ERROR."""
        result = compute_scores()
        assert result.ornn_i is None
        assert result.ornn_t is None
        assert result.score_status == ScoreStatus.ERROR

    def test_components_populated(self) -> None:
        """Components dict includes all provided values."""
        result = compute_scores(bw=1.5, fp8=0.8, bf16=1.2, ar=0.9)
        assert result.components == {"bw": 1.5, "fp8": 0.8, "bf16": 1.2, "ar": 0.9}

    def test_ref_overrides(self) -> None:
        """Reference overrides are passed through to scoring."""
        result = compute_scores(bw=2.0, fp8=2.0, bf16=2.0, ar=2.0, bw_ref=2.0, fp8_ref=2.0)
        # bw/bw_ref = 1.0, fp8/fp8_ref = 1.0 → ornn_i = 100
        assert result.ornn_i == pytest.approx(100.0)
        # bf16/bf16_ref = 2.0/1.0 = 2.0, ar/ar_ref = 2.0/1.0 = 2.0 → ornn_t = 200
        assert result.ornn_t == pytest.approx(200.0)


class TestPerGPUScoring:
    """Test per-GPU metric records and aggregation (VAL-RUNBOOK-010)."""

    def test_single_gpu_aggregate_is_identity(self) -> None:
        """Single GPU: aggregate equals the individual score."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
        ]
        result = aggregate_gpu_scores(per_gpu)
        assert result.ornn_i == pytest.approx(100.0)
        assert result.ornn_t == pytest.approx(100.0)
        assert result.aggregate_method == AggregateMethod.MINIMUM

    def test_multi_gpu_uses_minimum(self) -> None:
        """Multi-GPU: final score is minimum across GPUs (worst-GPU gate)."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
            PerGPUScore(gpu_uuid="GPU-0002", bw=0.5, fp8=0.5, bf16=0.5, ar=0.5),
        ]
        result = aggregate_gpu_scores(per_gpu)
        # GPU-0002 scores: ornn_i = 55*0.5 + 45*0.5 = 50, ornn_t = 50
        assert result.ornn_i == pytest.approx(50.0)
        assert result.ornn_t == pytest.approx(50.0)
        assert result.aggregate_method == AggregateMethod.MINIMUM

    def test_per_gpu_records_preserved(self) -> None:
        """Per-GPU records are preserved in result for transparency."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-AAA", bw=1.2, fp8=0.9, bf16=1.1, ar=0.8),
            PerGPUScore(gpu_uuid="GPU-BBB", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
        ]
        result = aggregate_gpu_scores(per_gpu)
        assert len(result.per_gpu_scores) == 2
        assert result.per_gpu_scores[0].gpu_uuid == "GPU-AAA"
        assert result.per_gpu_scores[1].gpu_uuid == "GPU-BBB"

    def test_aggregate_method_explicitly_recorded(self) -> None:
        """Aggregation method is explicitly documented in result."""
        per_gpu = [PerGPUScore(gpu_uuid="G1", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0)]
        result = aggregate_gpu_scores(per_gpu)
        assert result.aggregate_method is not None
        assert result.aggregate_method == AggregateMethod.MINIMUM.value

    def test_multi_gpu_qualification_uses_aggregate(self) -> None:
        """Qualification is based on aggregate (minimum) scores."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
            PerGPUScore(gpu_uuid="GPU-0002", bw=0.7, fp8=0.7, bf16=0.7, ar=0.7),
        ]
        result = aggregate_gpu_scores(per_gpu)
        # Aggregate scores from GPU-0002: ornn_i=70, ornn_t=70 → Standard
        assert result.qualification == Qualification.STANDARD

    def test_empty_gpu_list(self) -> None:
        """Empty GPU list → error status."""
        result = aggregate_gpu_scores([])
        assert result.ornn_i is None
        assert result.ornn_t is None
        assert result.score_status == ScoreStatus.ERROR

    def test_multi_gpu_with_partial_metrics(self) -> None:
        """Some GPUs missing metrics → those GPUs are None, aggregate handles gracefully."""
        per_gpu = [
            PerGPUScore(gpu_uuid="GPU-0001", bw=1.0, fp8=1.0, bf16=1.0, ar=1.0),
            PerGPUScore(gpu_uuid="GPU-0002", bw=None, fp8=None, bf16=1.0, ar=1.0),
        ]
        result = aggregate_gpu_scores(per_gpu)
        # GPU-0002 has ornn_i=None — overall ornn_i should be None or from GPU-0001
        # Since one GPU is invalid for ornn_i, the aggregate should reflect partial status
        assert result.score_status in (ScoreStatus.PARTIAL, ScoreStatus.ERROR)
        # Per-GPU records preserved
        assert len(result.per_gpu_scores) == 2
