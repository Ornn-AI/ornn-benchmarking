"""Scoring engine for Ornn-I and Ornn-T computation.

Provides deterministic local scoring with explicit status tracking,
multi-GPU aggregation semantics, and NaN/Inf/negative input rejection.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel

from ornn_bench.models import (
    PerGPUScoreRecord,
    Qualification,
    ScoreResult,
    ScoreStatus,
)

# Reference values for score normalization (placeholder defaults).
DEFAULT_REFS: dict[str, float] = {
    "bw_ref": 1.0,
    "fp8_ref": 1.0,
    "bf16_ref": 1.0,
    "ar_ref": 1.0,
}


class AggregateMethod(str, Enum):
    """Method used to aggregate per-GPU scores into a final score."""

    MINIMUM = "minimum"


class PerGPUScore(BaseModel):
    """Input metrics for a single GPU before scoring."""

    gpu_uuid: str
    bw: float | None = None
    fp8: float | None = None
    bf16: float | None = None
    ar: float | None = None


def _is_valid_metric(value: float | None) -> bool:
    """Check if a metric value is valid (not None, NaN, Inf, or negative)."""
    if value is None:
        return False
    if math.isnan(value) or math.isinf(value):
        return False
    return value >= 0


def _is_valid_ref(value: float) -> bool:
    """Check if a reference value is valid (positive and finite)."""
    if value <= 0:
        return False
    return not (math.isnan(value) or math.isinf(value))


def compute_ornn_i(
    bw: float | None,
    fp8: float | None,
    *,
    bw_ref: float = DEFAULT_REFS["bw_ref"],
    fp8_ref: float = DEFAULT_REFS["fp8_ref"],
) -> float | None:
    """Compute Ornn-I = 55*(BW/BW_ref) + 45*(FP8/FP8_ref).

    Returns None if required inputs are missing, invalid (NaN/Inf/negative),
    or if reference values are invalid.
    """
    if not _is_valid_metric(bw) or not _is_valid_metric(fp8):
        return None
    if not _is_valid_ref(bw_ref) or not _is_valid_ref(fp8_ref):
        return None
    assert bw is not None and fp8 is not None  # for type narrowing
    return 55.0 * (bw / bw_ref) + 45.0 * (fp8 / fp8_ref)


def compute_ornn_t(
    bf16: float | None,
    ar: float | None,
    *,
    bf16_ref: float = DEFAULT_REFS["bf16_ref"],
    ar_ref: float = DEFAULT_REFS["ar_ref"],
) -> float | None:
    """Compute Ornn-T = 55*(BF16/BF16_ref) + 45*(AR/AR_ref).

    Returns None if required inputs are missing, invalid (NaN/Inf/negative),
    or if reference values are invalid.
    """
    if not _is_valid_metric(bf16) or not _is_valid_metric(ar):
        return None
    if not _is_valid_ref(bf16_ref) or not _is_valid_ref(ar_ref):
        return None
    assert bf16 is not None and ar is not None  # for type narrowing
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


def _determine_score_status(
    ornn_i: float | None,
    ornn_t: float | None,
) -> tuple[ScoreStatus, str | None]:
    """Determine the explicit score status and detail message."""
    if ornn_i is not None and ornn_t is not None:
        return ScoreStatus.VALID, None
    if ornn_i is not None or ornn_t is not None:
        missing = []
        if ornn_i is None:
            missing.append("Ornn-I (missing/invalid inference metrics: BW, FP8)")
        if ornn_t is None:
            missing.append("Ornn-T (missing/invalid training metrics: BF16, AR)")
        detail = f"Partial score: missing {', '.join(missing)}"
        return ScoreStatus.PARTIAL, detail
    return ScoreStatus.ERROR, "No valid metrics available for scoring"


def compute_scores(
    bw: float | None = None,
    fp8: float | None = None,
    bf16: float | None = None,
    ar: float | None = None,
    **ref_overrides: float,
) -> ScoreResult:
    """Compute full score result with qualification and explicit status.

    Never produces NaN scores — invalid inputs result in None with
    explicit error/partial status.
    """
    refs = {**DEFAULT_REFS, **ref_overrides}

    ornn_i = compute_ornn_i(bw, fp8, bw_ref=refs["bw_ref"], fp8_ref=refs["fp8_ref"])
    ornn_t = compute_ornn_t(bf16, ar, bf16_ref=refs["bf16_ref"], ar_ref=refs["ar_ref"])
    qualification = determine_qualification(ornn_i, ornn_t)
    score_status, score_status_detail = _determine_score_status(ornn_i, ornn_t)

    components: dict[str, float] = {}
    for name, value in [("bw", bw), ("fp8", fp8), ("bf16", bf16), ("ar", ar)]:
        if value is not None and _is_valid_metric(value):
            components[name] = value

    return ScoreResult(
        ornn_i=ornn_i,
        ornn_t=ornn_t,
        qualification=qualification,
        components=components,
        score_status=score_status,
        score_status_detail=score_status_detail,
    )


def aggregate_gpu_scores(
    per_gpu: list[PerGPUScore],
    **ref_overrides: float,
) -> ScoreResult:
    """Aggregate per-GPU metrics into a final score using minimum-GPU semantics.

    For multi-GPU systems, the final Ornn-I and Ornn-T are the minimum
    across all GPUs. This ensures the weakest GPU gates qualification.

    The aggregation method is explicitly recorded in the result for
    transparency (VAL-RUNBOOK-010).

    Returns ScoreResult with per_gpu_scores and aggregate_method populated.
    """
    if not per_gpu:
        return ScoreResult(
            score_status=ScoreStatus.ERROR,
            score_status_detail="No GPU data available for scoring",
            aggregate_method=AggregateMethod.MINIMUM.value,
        )

    refs = {**DEFAULT_REFS, **ref_overrides}

    # Compute per-GPU scores
    gpu_records: list[PerGPUScoreRecord] = []
    ornn_i_values: list[float] = []
    ornn_t_values: list[float] = []

    for gpu in per_gpu:
        gpu_ornn_i = compute_ornn_i(
            gpu.bw, gpu.fp8, bw_ref=refs["bw_ref"], fp8_ref=refs["fp8_ref"]
        )
        gpu_ornn_t = compute_ornn_t(
            gpu.bf16, gpu.ar, bf16_ref=refs["bf16_ref"], ar_ref=refs["ar_ref"]
        )

        gpu_components: dict[str, float] = {}
        for name, value in [("bw", gpu.bw), ("fp8", gpu.fp8), ("bf16", gpu.bf16), ("ar", gpu.ar)]:
            if value is not None and _is_valid_metric(value):
                gpu_components[name] = value

        gpu_records.append(
            PerGPUScoreRecord(
                gpu_uuid=gpu.gpu_uuid,
                ornn_i=gpu_ornn_i,
                ornn_t=gpu_ornn_t,
                components=gpu_components,
            )
        )

        if gpu_ornn_i is not None:
            ornn_i_values.append(gpu_ornn_i)
        if gpu_ornn_t is not None:
            ornn_t_values.append(gpu_ornn_t)

    # Aggregate: minimum across valid GPU scores
    agg_ornn_i: float | None = min(ornn_i_values) if ornn_i_values else None
    agg_ornn_t: float | None = min(ornn_t_values) if ornn_t_values else None

    # Determine if some GPUs had invalid scores — this affects status
    total_gpus = len(per_gpu)
    valid_i_count = len(ornn_i_values)
    valid_t_count = len(ornn_t_values)

    # If not all GPUs contributed to both scores, status is partial
    detail: str | None
    if valid_i_count < total_gpus or valid_t_count < total_gpus:
        if agg_ornn_i is not None or agg_ornn_t is not None:
            status = ScoreStatus.PARTIAL
            detail_parts = []
            if valid_i_count < total_gpus:
                detail_parts.append(
                    f"Ornn-I: {valid_i_count}/{total_gpus} GPUs had valid inference metrics"
                )
            if valid_t_count < total_gpus:
                detail_parts.append(
                    f"Ornn-T: {valid_t_count}/{total_gpus} GPUs had valid training metrics"
                )
            detail = f"Partial aggregate: {'; '.join(detail_parts)}"
        else:
            status = ScoreStatus.ERROR
            detail = "No GPUs produced valid scores"
    else:
        status, detail = _determine_score_status(agg_ornn_i, agg_ornn_t)

    qualification = determine_qualification(agg_ornn_i, agg_ornn_t)

    # Collect aggregate components from the minimum-scoring GPU for each metric
    agg_components: dict[str, float] = {}
    for record in gpu_records:
        for key, val in record.components.items():
            if key not in agg_components or val < agg_components[key]:
                agg_components[key] = val

    return ScoreResult(
        ornn_i=agg_ornn_i,
        ornn_t=agg_ornn_t,
        qualification=qualification,
        components=agg_components,
        score_status=status,
        score_status_detail=detail,
        aggregate_method=AggregateMethod.MINIMUM.value,
        per_gpu_scores=gpu_records,
    )
