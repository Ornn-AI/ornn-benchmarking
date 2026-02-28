"""Tests for the scoring engine."""

from __future__ import annotations

import pytest

from ornn_bench.models import Qualification
from ornn_bench.scoring import (
    compute_ornn_i,
    compute_ornn_t,
    compute_scores,
    determine_qualification,
)


class TestComputeOrnnI:
    """Tests for Ornn-I computation."""

    def test_basic_computation(self) -> None:
        """Ornn-I = 55*(BW/BW_ref) + 45*(FP8/FP8_ref)."""
        result = compute_ornn_i(1.0, 1.0, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(100.0)

    def test_with_different_values(self) -> None:
        """Verify formula with non-unit values."""
        result = compute_ornn_i(2.0, 0.5, bw_ref=1.0, fp8_ref=1.0)
        assert result == pytest.approx(55.0 * 2.0 + 45.0 * 0.5)

    def test_returns_none_when_bw_missing(self) -> None:
        """Returns None if BW is missing."""
        assert compute_ornn_i(None, 1.0) is None

    def test_returns_none_when_fp8_missing(self) -> None:
        """Returns None if FP8 is missing."""
        assert compute_ornn_i(1.0, None) is None

    def test_returns_none_when_ref_zero(self) -> None:
        """Returns None if reference value is zero."""
        assert compute_ornn_i(1.0, 1.0, bw_ref=0.0) is None

    def test_returns_none_when_ref_negative(self) -> None:
        """Returns None if reference value is negative."""
        assert compute_ornn_i(1.0, 1.0, fp8_ref=-1.0) is None


class TestComputeOrnnT:
    """Tests for Ornn-T computation."""

    def test_basic_computation(self) -> None:
        """Ornn-T = 55*(BF16/BF16_ref) + 45*(AR/AR_ref)."""
        result = compute_ornn_t(1.0, 1.0, bf16_ref=1.0, ar_ref=1.0)
        assert result == pytest.approx(100.0)

    def test_with_different_values(self) -> None:
        """Verify formula with non-unit values."""
        result = compute_ornn_t(0.8, 1.5, bf16_ref=1.0, ar_ref=1.0)
        assert result == pytest.approx(55.0 * 0.8 + 45.0 * 1.5)

    def test_returns_none_when_bf16_missing(self) -> None:
        """Returns None if BF16 is missing."""
        assert compute_ornn_t(None, 1.0) is None

    def test_returns_none_when_ar_missing(self) -> None:
        """Returns None if AR is missing."""
        assert compute_ornn_t(1.0, None) is None


class TestDetermineQualification:
    """Tests for qualification grade determination."""

    def test_premium_qualification(self) -> None:
        """Premium: composite >= 90 and both floors >= 80."""
        result = determine_qualification(95.0, 95.0)
        assert result == Qualification.PREMIUM

    def test_premium_at_boundary(self) -> None:
        """Premium boundary: composite=90, floors=80."""
        result = determine_qualification(90.0, 90.0)
        assert result == Qualification.PREMIUM

    def test_premium_fails_floor(self) -> None:
        """Composite >= 90 but one floor < 80 → not Premium."""
        result = determine_qualification(100.0, 79.0)
        assert result != Qualification.PREMIUM

    def test_standard_qualification(self) -> None:
        """Standard: composite >= 70 and both floors >= 60."""
        result = determine_qualification(75.0, 75.0)
        assert result == Qualification.STANDARD

    def test_standard_at_boundary(self) -> None:
        """Standard boundary: composite=70, floors=60."""
        result = determine_qualification(70.0, 70.0)
        assert result == Qualification.STANDARD

    def test_below_qualification(self) -> None:
        """Below: composite < 70 or a floor < 60."""
        result = determine_qualification(50.0, 50.0)
        assert result == Qualification.BELOW

    def test_none_when_scores_missing(self) -> None:
        """Returns None if either score is missing."""
        assert determine_qualification(None, 90.0) is None
        assert determine_qualification(90.0, None) is None


class TestComputeScores:
    """Tests for the aggregate score computation."""

    def test_full_computation(self) -> None:
        """Compute full scores with all inputs."""
        result = compute_scores(bw=1.0, fp8=1.0, bf16=1.0, ar=1.0)
        assert result.ornn_i == pytest.approx(100.0)
        assert result.ornn_t == pytest.approx(100.0)
        assert result.qualification == Qualification.PREMIUM

    def test_partial_computation(self) -> None:
        """Partial inputs produce partial scores."""
        result = compute_scores(bw=1.0, fp8=1.0)
        assert result.ornn_i == pytest.approx(100.0)
        assert result.ornn_t is None
        assert result.qualification is None

    def test_no_inputs(self) -> None:
        """No inputs produce no scores."""
        result = compute_scores()
        assert result.ornn_i is None
        assert result.ornn_t is None
        assert result.qualification is None

    def test_components_populated(self) -> None:
        """Components dict includes provided values."""
        result = compute_scores(bw=1.5, fp8=0.8, bf16=1.2)
        assert result.components["bw"] == 1.5
        assert result.components["fp8"] == 0.8
        assert result.components["bf16"] == 1.2
        assert "ar" not in result.components
