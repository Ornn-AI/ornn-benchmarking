"""Tests for server-side scoring engine — formula correctness and qualification.

Covers:
  - VAL-API-007: Server score formula correctness
  - VAL-API-008: Qualification rule correctness
"""

from __future__ import annotations

import pytest
from api.scoring import (
    DEFAULT_TOLERANCE,
    Qualification,
    VerificationStatus,
    compute_ornn_i,
    compute_ornn_t,
    determine_qualification,
    recompute_and_verify,
)

# ---------------------------------------------------------------------------
# VAL-API-007: Server score formula correctness
# ---------------------------------------------------------------------------


class TestServerOrnnI:
    """Server-side Ornn-I formula: 55*(BW/BW_ref) + 45*(FP8/FP8_ref)."""

    def test_unit_reference_values(self) -> None:
        """All metrics at reference → score = 100."""
        result = compute_ornn_i(1.0, 1.0, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(100.0)

    def test_double_bw(self) -> None:
        """BW=2x reference, FP8=1x → 55*2 + 45*1 = 155."""
        result = compute_ornn_i(2.0, 1.0, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(155.0)

    def test_half_fp8(self) -> None:
        """BW=1x, FP8=0.5x → 55*1 + 45*0.5 = 77.5."""
        result = compute_ornn_i(1.0, 0.5, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(77.5)

    def test_custom_reference_values(self) -> None:
        """Custom references scale correctly."""
        # bw=200, bw_ref=100 → ratio=2, fp8=450, fp8_ref=500 → ratio=0.9
        result = compute_ornn_i(200.0, 450.0, bw_ref=100.0, fp8_ref=500.0)
        expected = 55.0 * 2.0 + 45.0 * 0.9
        assert result == pytest.approx(expected)

    def test_none_bw_returns_none(self) -> None:
        """Missing BW → None."""
        assert compute_ornn_i(None, 1.0) is None

    def test_none_fp8_returns_none(self) -> None:
        """Missing FP8 → None."""
        assert compute_ornn_i(1.0, None) is None

    def test_zero_ref_returns_none(self) -> None:
        """Zero reference → None (avoid division by zero)."""
        assert compute_ornn_i(1.0, 1.0, bw_ref=0.0) is None

    def test_negative_ref_returns_none(self) -> None:
        """Negative reference → None."""
        assert compute_ornn_i(1.0, 1.0, fp8_ref=-1.0) is None

    def test_nan_input_returns_none(self) -> None:
        """NaN input → None."""
        assert compute_ornn_i(float("nan"), 1.0) is None

    def test_inf_input_returns_none(self) -> None:
        """Inf input → None."""
        assert compute_ornn_i(1.0, float("inf")) is None

    def test_negative_metric_returns_none(self) -> None:
        """Negative metric → None."""
        assert compute_ornn_i(-1.0, 1.0) is None

    def test_zero_metric_is_valid(self) -> None:
        """Zero metric is valid (GPU could genuinely measure 0)."""
        result = compute_ornn_i(0.0, 1.0, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(45.0)  # 55*0 + 45*1


class TestServerOrnnT:
    """Server-side Ornn-T formula: 55*(BF16/BF16_ref) + 45*(AR/AR_ref)."""

    def test_unit_reference_values(self) -> None:
        """All metrics at reference → score = 100."""
        result = compute_ornn_t(1.0, 1.0, bf16_ref=1.0, ar_ref=1.0)
        assert result == pytest.approx(100.0)

    def test_double_bf16(self) -> None:
        """BF16=2x, AR=1x → 55*2 + 45*1 = 155."""
        result = compute_ornn_t(2.0, 1.0, bf16_ref=1.0, ar_ref=1.0)
        assert result == pytest.approx(155.0)

    def test_half_ar(self) -> None:
        """BF16=1x, AR=0.5x → 55*1 + 45*0.5 = 77.5."""
        result = compute_ornn_t(1.0, 0.5, bf16_ref=1.0, ar_ref=1.0)
        assert result == pytest.approx(77.5)

    def test_custom_reference_values(self) -> None:
        """Custom references scale correctly."""
        result = compute_ornn_t(300.0, 150.0, bf16_ref=200.0, ar_ref=100.0)
        expected = 55.0 * (300.0 / 200.0) + 45.0 * (150.0 / 100.0)
        assert result == pytest.approx(expected)

    def test_none_bf16_returns_none(self) -> None:
        """Missing BF16 → None."""
        assert compute_ornn_t(None, 1.0) is None

    def test_none_ar_returns_none(self) -> None:
        """Missing AR → None."""
        assert compute_ornn_t(1.0, None) is None

    def test_zero_ref_returns_none(self) -> None:
        """Zero reference → None."""
        assert compute_ornn_t(1.0, 1.0, bf16_ref=0.0) is None

    def test_nan_input_returns_none(self) -> None:
        """NaN input → None."""
        assert compute_ornn_t(1.0, float("nan")) is None


# ---------------------------------------------------------------------------
# Deterministic fixture vectors (VAL-API-007 test vectors)
# ---------------------------------------------------------------------------


class TestDeterministicFixtureVectors:
    """Server formula matches expected values for deterministic test vectors."""

    @pytest.mark.parametrize(
        "bw, fp8, bw_ref, fp8_ref, expected_i",
        [
            # Vector 1: Reference baseline
            (1.0, 1.0, 1.0, 1.0, 100.0),
            # Vector 2: 2x BW, 0.5x FP8
            (2.0, 0.5, 1.0, 1.0, 132.5),
            # Vector 3: Real-world-ish values
            (3200.0, 800.0, 3000.0, 750.0, 55.0 * (3200.0 / 3000.0) + 45.0 * (800.0 / 750.0)),
            # Vector 4: All zero metrics
            (0.0, 0.0, 1.0, 1.0, 0.0),
            # Vector 5: Extremely high performance
            (10.0, 10.0, 1.0, 1.0, 1000.0),
        ],
    )
    def test_ornn_i_fixture_vector(
        self,
        bw: float,
        fp8: float,
        bw_ref: float,
        fp8_ref: float,
        expected_i: float,
    ) -> None:
        """Ornn-I fixture vector matches expected value."""
        result = compute_ornn_i(bw, fp8, bw_ref=bw_ref, fp8_ref=fp8_ref)
        assert result == pytest.approx(expected_i)

    @pytest.mark.parametrize(
        "bf16, ar, bf16_ref, ar_ref, expected_t",
        [
            # Vector 1: Reference baseline
            (1.0, 1.0, 1.0, 1.0, 100.0),
            # Vector 2: 0.8x BF16, 1.5x AR
            (0.8, 1.5, 1.0, 1.0, 55.0 * 0.8 + 45.0 * 1.5),
            # Vector 3: Real-world-ish values
            (500.0, 200.0, 400.0, 180.0, 55.0 * (500.0 / 400.0) + 45.0 * (200.0 / 180.0)),
            # Vector 4: All zero
            (0.0, 0.0, 1.0, 1.0, 0.0),
        ],
    )
    def test_ornn_t_fixture_vector(
        self,
        bf16: float,
        ar: float,
        bf16_ref: float,
        ar_ref: float,
        expected_t: float,
    ) -> None:
        """Ornn-T fixture vector matches expected value."""
        result = compute_ornn_t(bf16, ar, bf16_ref=bf16_ref, ar_ref=ar_ref)
        assert result == pytest.approx(expected_t)


# ---------------------------------------------------------------------------
# VAL-API-008: Qualification rule correctness
# ---------------------------------------------------------------------------


class TestQualificationRules:
    """Premium/Standard/Below qualification with floor + composite gate."""

    def test_premium_high_scores(self) -> None:
        """Both scores high → Premium."""
        result = determine_qualification(95.0, 95.0)
        assert result == Qualification.PREMIUM.value

    def test_premium_at_exact_boundary(self) -> None:
        """Composite=90, both floors=80 → Premium."""
        # ornn_i=80, ornn_t=100 → composite=(80+100)/2=90
        result = determine_qualification(80.0, 100.0)
        assert result == Qualification.PREMIUM.value

    def test_premium_boundary_floors_exact(self) -> None:
        """Both at exactly 90 → composite=90, floors=90 → Premium."""
        result = determine_qualification(90.0, 90.0)
        assert result == Qualification.PREMIUM.value

    def test_premium_fails_floor_low_i(self) -> None:
        """Composite >= 90 but ornn_i < 80 → not Premium."""
        # ornn_i=79, ornn_t=101 → composite=90, but floor fails
        result = determine_qualification(79.0, 101.0)
        assert result != Qualification.PREMIUM.value

    def test_premium_fails_floor_low_t(self) -> None:
        """Composite >= 90 but ornn_t < 80 → not Premium."""
        result = determine_qualification(101.0, 79.0)
        assert result != Qualification.PREMIUM.value

    def test_standard_mid_scores(self) -> None:
        """Mid-range scores → Standard."""
        result = determine_qualification(75.0, 75.0)
        assert result == Qualification.STANDARD.value

    def test_standard_at_exact_boundary(self) -> None:
        """Composite=70, both floors=60 → Standard."""
        # ornn_i=60, ornn_t=80 → composite=70, floors: 60, 80 → ok
        result = determine_qualification(60.0, 80.0)
        assert result == Qualification.STANDARD.value

    def test_standard_boundary_exact(self) -> None:
        """Both at 70 → composite=70, floors=70 → Standard."""
        result = determine_qualification(70.0, 70.0)
        assert result == Qualification.STANDARD.value

    def test_standard_fails_floor_low_i(self) -> None:
        """Composite >= 70 but ornn_i < 60 → Below."""
        # ornn_i=59, ornn_t=81 → composite=70, floor fails
        result = determine_qualification(59.0, 81.0)
        assert result == Qualification.BELOW.value

    def test_below_low_scores(self) -> None:
        """Low scores → Below."""
        result = determine_qualification(50.0, 50.0)
        assert result == Qualification.BELOW.value

    def test_below_very_low(self) -> None:
        """Very low scores → Below."""
        result = determine_qualification(10.0, 10.0)
        assert result == Qualification.BELOW.value

    def test_none_when_i_missing(self) -> None:
        """None ornn_i → None."""
        assert determine_qualification(None, 90.0) is None

    def test_none_when_t_missing(self) -> None:
        """None ornn_t → None."""
        assert determine_qualification(90.0, None) is None

    def test_none_when_both_missing(self) -> None:
        """Both None → None."""
        assert determine_qualification(None, None) is None


# ---------------------------------------------------------------------------
# Threshold boundary fixture vectors (VAL-API-008)
# ---------------------------------------------------------------------------


class TestQualificationBoundaryVectors:
    """Parametrized threshold boundary tests for qualification gate."""

    @pytest.mark.parametrize(
        "ornn_i, ornn_t, expected_qual",
        [
            # Premium boundary cases
            (90.0, 90.0, "Premium"),      # exact boundary → Premium
            (80.0, 100.0, "Premium"),     # composite=90, floor=80 → Premium
            (100.0, 80.0, "Premium"),     # composite=90, floor=80 → Premium
            (89.9, 90.1, "Premium"),      # composite=90, both >= 80 → Premium
            (79.9, 100.1, "Standard"),    # composite=90, but i floor=79.9 < 80 → not Premium
            (100.1, 79.9, "Standard"),    # composite=90, but t floor=79.9 < 80 → not Premium
            # Standard boundary cases
            (70.0, 70.0, "Standard"),     # exact boundary → Standard
            (60.0, 80.0, "Standard"),     # composite=70, floor=60 → Standard
            (80.0, 60.0, "Standard"),     # composite=70, floor=60 → Standard
            (59.9, 80.1, "Below"),        # composite=70, floor=59.9 < 60 → Below
            (80.1, 59.9, "Below"),        # composite=70, floor=59.9 < 60 → Below
            # Below cases
            (50.0, 50.0, "Below"),
            (0.0, 0.0, "Below"),
            (69.9, 69.9, "Below"),        # composite=69.9, both >= 60 but composite < 70
        ],
    )
    def test_qualification_boundary(
        self,
        ornn_i: float,
        ornn_t: float,
        expected_qual: str,
    ) -> None:
        """Qualification matches expected value at boundary."""
        result = determine_qualification(ornn_i, ornn_t)
        assert result == expected_qual


# ---------------------------------------------------------------------------
# Recompute-and-verify integration tests
# ---------------------------------------------------------------------------


class TestRecomputeAndVerify:
    """Integration tests for the full recompute_and_verify flow."""

    def test_perfect_match_verified(self) -> None:
        """Matching components and scores → verified."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=100.0,
            submitted_ornn_t=100.0,
            submitted_qualification="Premium",
        )
        assert result.status == VerificationStatus.VERIFIED
        assert result.server_ornn_i == pytest.approx(100.0)
        assert result.server_ornn_t == pytest.approx(100.0)

    def test_score_mismatch_detected(self) -> None:
        """Wrong submitted score → mismatch."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=999.0,
            submitted_ornn_t=100.0,
            submitted_qualification="Premium",
        )
        assert result.status == VerificationStatus.MISMATCH

    def test_qualification_mismatch(self) -> None:
        """Correct scores but wrong qualification → mismatch."""
        result = recompute_and_verify(
            components={"bw": 0.5, "fp8": 0.5, "bf16": 0.5, "ar": 0.5},
            submitted_ornn_i=50.0,
            submitted_ornn_t=50.0,
            submitted_qualification="Premium",  # should be Below
        )
        assert result.status == VerificationStatus.MISMATCH

    def test_partial_components_none_scores_verified(self) -> None:
        """Partial components with None submitted → verified."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0},
            submitted_ornn_i=100.0,
            submitted_ornn_t=None,
            submitted_qualification=None,
        )
        assert result.status == VerificationStatus.VERIFIED

    def test_empty_components_none_scores_verified(self) -> None:
        """Empty components with None scores → verified (consistent)."""
        result = recompute_and_verify(
            components={},
            submitted_ornn_i=None,
            submitted_ornn_t=None,
            submitted_qualification=None,
        )
        assert result.status == VerificationStatus.VERIFIED

    def test_metric_details_count(self) -> None:
        """Metric details include ornn_i, ornn_t, qualification."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=100.0,
            submitted_ornn_t=100.0,
            submitted_qualification="Premium",
        )
        metric_names = [d.metric for d in result.metric_details]
        assert "ornn_i" in metric_names
        assert "ornn_t" in metric_names
        assert "qualification" in metric_names

    def test_tolerance_applied(self) -> None:
        """Small delta within tolerance → verified."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=100.005,
            submitted_ornn_t=99.995,
            submitted_qualification="Premium",
        )
        assert result.status == VerificationStatus.VERIFIED
        assert result.tolerance == DEFAULT_TOLERANCE

    def test_beyond_tolerance_is_mismatch(self) -> None:
        """Delta exceeding tolerance → mismatch."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=100.02,  # delta=0.02 > default 0.01
            submitted_ornn_t=100.0,
            submitted_qualification="Premium",
        )
        assert result.status == VerificationStatus.MISMATCH

    def test_custom_tolerance(self) -> None:
        """Custom tolerance allows larger deltas."""
        result = recompute_and_verify(
            components={"bw": 1.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=100.05,
            submitted_ornn_t=100.0,
            submitted_qualification="Premium",
            tolerance=0.1,
        )
        assert result.status == VerificationStatus.VERIFIED

    def test_ref_overrides(self) -> None:
        """Reference overrides change the expected score."""
        # bw=2, fp8=1 with bw_ref=2, fp8_ref=1 → ratios=1 → score=100
        result = recompute_and_verify(
            components={"bw": 2.0, "fp8": 1.0, "bf16": 1.0, "ar": 1.0},
            submitted_ornn_i=100.0,
            submitted_ornn_t=100.0,
            submitted_qualification="Premium",
            ref_overrides={"bw_ref": 2.0, "fp8_ref": 1.0},
        )
        assert result.status == VerificationStatus.VERIFIED
        assert result.server_ornn_i == pytest.approx(100.0)
