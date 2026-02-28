"""Scoring engine for Ornn-I and Ornn-T computation."""

from __future__ import annotations

from ornn_bench.models import Qualification, ScoreResult

# Reference values for score normalization (placeholder defaults).
DEFAULT_REFS: dict[str, float] = {
    "bw_ref": 1.0,
    "fp8_ref": 1.0,
    "bf16_ref": 1.0,
    "ar_ref": 1.0,
}


def compute_ornn_i(
    bw: float | None,
    fp8: float | None,
    *,
    bw_ref: float = DEFAULT_REFS["bw_ref"],
    fp8_ref: float = DEFAULT_REFS["fp8_ref"],
) -> float | None:
    """Compute Ornn-I = 55*(BW/BW_ref) + 45*(FP8/FP8_ref).

    Returns None if required inputs are missing.
    """
    if bw is None or fp8 is None:
        return None
    if bw_ref <= 0 or fp8_ref <= 0:
        return None
    return 55.0 * (bw / bw_ref) + 45.0 * (fp8 / fp8_ref)


def compute_ornn_t(
    bf16: float | None,
    ar: float | None,
    *,
    bf16_ref: float = DEFAULT_REFS["bf16_ref"],
    ar_ref: float = DEFAULT_REFS["ar_ref"],
) -> float | None:
    """Compute Ornn-T = 55*(BF16/BF16_ref) + 45*(AR/AR_ref).

    Returns None if required inputs are missing.
    """
    if bf16 is None or ar is None:
        return None
    if bf16_ref <= 0 or ar_ref <= 0:
        return None
    return 55.0 * (bf16 / bf16_ref) + 45.0 * (ar / ar_ref)


def determine_qualification(
    ornn_i: float | None,
    ornn_t: float | None,
) -> Qualification | None:
    """Determine qualification grade based on scores.

    Applies floor + composite gate logic.
    Returns None if scores are missing.
    """
    if ornn_i is None or ornn_t is None:
        return None
    composite = (ornn_i + ornn_t) / 2.0
    if composite >= 90.0 and ornn_i >= 80.0 and ornn_t >= 80.0:
        return Qualification.PREMIUM
    if composite >= 70.0 and ornn_i >= 60.0 and ornn_t >= 60.0:
        return Qualification.STANDARD
    return Qualification.BELOW


def compute_scores(
    bw: float | None = None,
    fp8: float | None = None,
    bf16: float | None = None,
    ar: float | None = None,
    **ref_overrides: float,
) -> ScoreResult:
    """Compute full score result with qualification."""
    refs = {**DEFAULT_REFS, **ref_overrides}

    ornn_i = compute_ornn_i(bw, fp8, bw_ref=refs["bw_ref"], fp8_ref=refs["fp8_ref"])
    ornn_t = compute_ornn_t(bf16, ar, bf16_ref=refs["bf16_ref"], ar_ref=refs["ar_ref"])
    qualification = determine_qualification(ornn_i, ornn_t)

    components: dict[str, float] = {}
    if bw is not None:
        components["bw"] = bw
    if fp8 is not None:
        components["fp8"] = fp8
    if bf16 is not None:
        components["bf16"] = bf16
    if ar is not None:
        components["ar"] = ar

    return ScoreResult(
        ornn_i=ornn_i,
        ornn_t=ornn_t,
        qualification=qualification,
        components=components,
    )
