"""Scoring engine for Ornn-I and Ornn-T computation.

Provides deterministic local scoring with explicit status tracking,
multi-GPU aggregation semantics, and NaN/Inf/negative input rejection.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel

from ornn_bench.models import (
    BenchmarkStatus,
    PerGPUScoreRecord,
    Qualification,
    ScoreResult,
    ScoreStatus,
    SectionResult,
)

# Reference values for score normalization (placeholder defaults).
DEFAULT_REFS: dict[str, float] = {
    "bw_ref": 1.0,
    "fp8_ref": 1.0,
    "bf16_ref": 1.0,
    "ar_ref": 1.0,
}

# Reference H100 SXM5 metrics used to normalize raw section outputs into ratios.
# These ratios are then scored locally with refs=1.0 so server-side verify can
# recompute from the submitted components and match exactly.
SECTION_METRIC_REFS: dict[str, float] = {
    "bw": 2039.5,
    "fp8": 1782.6,
    "bf16": 891.3,
    "ar": 148.32,
}

FP8_DTYPE_KEYS: tuple[str, ...] = ("fp8_e4m3", "fp8_e5m2")

BW_COMPONENT_SOURCE = "memory.nvbandwidth_results.device_local_copy.max"
FP8_COMPONENT_SOURCES: tuple[str, ...] = (
    "compute.per_gpu.<gpu>.fp8_e4m3.best.tflops",
    "compute.per_gpu.<gpu>.fp8_e5m2.best.tflops",
)
BF16_COMPONENT_SOURCE = "compute.per_gpu.<gpu>.bf16.best.tflops"
AR_COMPONENT_SOURCE = "interconnect.bus_bandwidth_summary.all_reduce_1gb.avg_busbw"


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


def _as_dict(value: object) -> dict[str, object]:
    """Return a shallow string-keyed dict when possible."""
    if not isinstance(value, dict):
        return {}
    return {str(key): val for key, val in value.items()}


def _as_list_of_dicts(value: object) -> list[dict[str, object]]:
    """Return only dict items from a list-like value."""
    if not isinstance(value, list):
        return []
    items: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            items.append({str(key): val for key, val in item.items()})
    return items


def _as_float(value: object) -> float | None:
    """Coerce a numeric-ish value to float when possible."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _find_section(sections: list[SectionResult], name: str) -> SectionResult | None:
    """Return the section result with the matching name, if present."""
    for section in sections:
        if section.name == name:
            return section
    return None


def _normalize_metric(value: float | None, reference: float) -> float | None:
    """Normalize a raw metric against its reference anchor."""
    if not _is_valid_metric(value) or not _is_valid_ref(reference):
        return None
    assert value is not None
    return value / reference


def _best_tflops(metric: object) -> float | None:
    """Extract ``best.tflops`` from a parsed MAMF result."""
    best = _as_dict(_as_dict(metric).get("best"))
    value = _as_float(best.get("tflops"))
    if value is None or not _is_valid_metric(value):
        return None
    return value


def _extract_metric_error(metric: object) -> str | None:
    """Extract an error string from a parsed metric dict when present."""
    error = _as_dict(metric).get("error")
    if isinstance(error, str) and error:
        return error
    return None


def _extract_gpu_uuids(preflight: SectionResult | None) -> list[str]:
    """Extract ordered GPU UUIDs from pre-flight inventory."""
    if preflight is None or preflight.status != BenchmarkStatus.COMPLETED:
        return []

    metrics = _as_dict(preflight.metrics)
    gpu_inventory = _as_dict(metrics.get("gpu_inventory"))
    gpu_entries = _as_list_of_dicts(gpu_inventory.get("gpus"))
    if not gpu_entries:
        gpu_entries = _as_list_of_dicts(metrics.get("gpus"))

    uuids: list[str] = []
    for index, gpu in enumerate(gpu_entries):
        uuid = gpu.get("uuid")
        if isinstance(uuid, str) and uuid:
            uuids.append(uuid)
        else:
            uuids.append(f"gpu_{index}")
    return uuids


def _extract_memory_bw_ratio(memory: SectionResult | None) -> tuple[float | None, list[str]]:
    """Extract normalized BW ratio from the memory section."""
    if memory is None:
        return None, [
            f"bw unavailable: missing memory section for {BW_COMPONENT_SOURCE}"
        ]
    if memory.status != BenchmarkStatus.COMPLETED:
        suffix = f": {memory.error}" if memory.error else ""
        return None, [
            f"bw unavailable: memory section {memory.status.value}{suffix}"
        ]

    metrics = _as_dict(memory.metrics)
    nvbandwidth_results = _as_dict(metrics.get("nvbandwidth_results"))
    device_local_copy = _as_dict(nvbandwidth_results.get("device_local_copy"))
    error = device_local_copy.get("error")
    if isinstance(error, str) and error:
        return None, [
            f"bw unavailable: {BW_COMPONENT_SOURCE} failed: {error}"
        ]
    raw_bw = _as_float(device_local_copy.get("max"))
    bw_ratio = _normalize_metric(raw_bw, SECTION_METRIC_REFS["bw"])
    if bw_ratio is None:
        return None, [
            f"bw unavailable: missing/invalid {BW_COMPONENT_SOURCE}"
        ]
    return bw_ratio, []


def _extract_ar_ratio(
    interconnect: SectionResult | None,
) -> tuple[float | None, list[str]]:
    """Extract normalized AR ratio from the interconnect section."""
    if interconnect is None:
        return None, [
            f"ar unavailable: missing interconnect section for {AR_COMPONENT_SOURCE}"
        ]
    if interconnect.status != BenchmarkStatus.COMPLETED:
        suffix = f": {interconnect.error}" if interconnect.error else ""
        return None, [
            f"ar unavailable: interconnect section {interconnect.status.value}{suffix}"
        ]

    metrics = _as_dict(interconnect.metrics)
    bus_bandwidth_summary = _as_dict(metrics.get("bus_bandwidth_summary"))
    all_reduce_1gb_summary = _as_dict(bus_bandwidth_summary.get("all_reduce_1gb"))
    nccl_results = _as_dict(metrics.get("nccl_results"))
    all_reduce_1gb = _as_dict(nccl_results.get("all_reduce_1gb"))
    error = all_reduce_1gb.get("error")
    if isinstance(error, str) and error:
        return None, [
            f"ar unavailable: {AR_COMPONENT_SOURCE} failed: {error}"
        ]
    raw_ar = _as_float(all_reduce_1gb_summary.get("avg_busbw"))

    if raw_ar is None:
        raw_ar = _as_float(all_reduce_1gb.get("avg_bus_bandwidth"))

    ar_ratio = _normalize_metric(raw_ar, SECTION_METRIC_REFS["ar"])
    if ar_ratio is None:
        return None, [
            f"ar unavailable: missing/invalid {AR_COMPONENT_SOURCE}"
        ]
    return ar_ratio, []


def _gpu_uuid_for_index(gpu_index: int, gpu_uuids: list[str]) -> str:
    """Return a GPU UUID fallback for the given index."""
    if 0 <= gpu_index < len(gpu_uuids):
        return gpu_uuids[gpu_index]
    return f"gpu_{gpu_index}"


def _expected_gpu_count(
    gpu_uuids: list[str],
    compute_per_gpu: dict[str, object],
    compute_metrics: dict[str, object],
) -> int:
    """Resolve the number of GPUs expected in the score output."""
    compute_gpu_count = _as_float(compute_metrics.get("gpu_count"))
    numeric_count = int(compute_gpu_count) if compute_gpu_count is not None else 0
    return max(len(gpu_uuids), numeric_count, len(compute_per_gpu))


def _extract_fp8_ratio_for_gpu(
    gpu_metrics: dict[str, object],
    gpu_label: str,
) -> tuple[float | None, list[str]]:
    """Extract normalized FP8 ratio from available FP8 dtype results."""
    raw_candidates: list[float] = []
    errors: list[str] = []
    for dtype in FP8_DTYPE_KEYS:
        metric = gpu_metrics.get(dtype)
        if (raw_value := _best_tflops(metric)) is not None:
            raw_candidates.append(raw_value)
            continue
        if error := _extract_metric_error(metric):
            errors.append(f"{dtype} failed: {error}")

    if not raw_candidates:
        suffix = f" ({'; '.join(errors)})" if errors else ""
        return None, [
            (
                "fp8 unavailable for "
                f"{gpu_label}: missing/invalid {' or '.join(FP8_COMPONENT_SOURCES)}{suffix}"
            )
        ]

    fp8_ratio = _normalize_metric(max(raw_candidates), SECTION_METRIC_REFS["fp8"])
    if fp8_ratio is None:
        return None, [
            (
                "fp8 unavailable for "
                f"{gpu_label}: missing/invalid {' or '.join(FP8_COMPONENT_SOURCES)}"
            )
        ]
    return fp8_ratio, []


def _extract_bf16_ratio_for_gpu(
    gpu_metrics: dict[str, object],
    gpu_label: str,
) -> tuple[float | None, list[str]]:
    """Extract normalized BF16 ratio from BF16 MAMF results."""
    bf16_metric = gpu_metrics.get("bf16")
    raw_bf16 = _best_tflops(bf16_metric)
    bf16_ratio = _normalize_metric(raw_bf16, SECTION_METRIC_REFS["bf16"])
    if bf16_ratio is None:
        error = _extract_metric_error(bf16_metric)
        suffix = f" ({error})" if error else ""
        return None, [
            f"bf16 unavailable for {gpu_label}: missing/invalid {BF16_COMPONENT_SOURCE}{suffix}"
        ]
    return bf16_ratio, []


def _build_score_detail(
    status: ScoreStatus,
    base_detail: str | None,
    issues: list[str],
) -> str | None:
    """Combine aggregate detail with source-level extraction issues."""
    ordered_issues = list(dict.fromkeys(issue for issue in issues if issue))
    if not ordered_issues:
        return base_detail

    issues_text = "; ".join(ordered_issues)
    if base_detail:
        return f"{base_detail}; source issues: {issues_text}"
    if status == ScoreStatus.PARTIAL:
        return f"Partial score: {issues_text}"
    if status == ScoreStatus.ERROR:
        return f"Scoring failed: {issues_text}"
    return issues_text


def derive_scores_from_sections(sections: list[SectionResult]) -> ScoreResult:
    """Derive normalized components and report scores from runbook sections.

    Raw runbook metrics are normalized against H100 SXM5 reference values so the
    persisted report contains ratio-based components. Those ratios are then used
    to compute per-GPU Ornn-I/Ornn-T scores and a minimum-GPU aggregate that the
    server can recompute with refs=1.0.
    """
    preflight = _find_section(sections, "pre-flight")
    compute = _find_section(sections, "compute")
    memory = _find_section(sections, "memory")
    interconnect = _find_section(sections, "interconnect")

    issues: list[str] = []
    gpu_uuids = _extract_gpu_uuids(preflight)
    bw_ratio, bw_issues = _extract_memory_bw_ratio(memory)
    ar_ratio, ar_issues = _extract_ar_ratio(interconnect)
    issues.extend(bw_issues)
    issues.extend(ar_issues)

    compute_metrics = (
        _as_dict(compute.metrics)
        if compute is not None and compute.status == BenchmarkStatus.COMPLETED
        else {}
    )
    compute_per_gpu = _as_dict(compute_metrics.get("per_gpu"))
    expected_gpu_count = _expected_gpu_count(gpu_uuids, compute_per_gpu, compute_metrics)

    if compute is None:
        issues.append("compute unavailable: missing compute section")
    elif compute.status != BenchmarkStatus.COMPLETED:
        suffix = f": {compute.error}" if compute.error else ""
        issues.append(
            f"compute unavailable: compute section {compute.status.value}{suffix}"
        )

    per_gpu_inputs: list[PerGPUScore] = []
    for gpu_index in range(expected_gpu_count):
        gpu_key = f"gpu_{gpu_index}"
        gpu_uuid = _gpu_uuid_for_index(gpu_index, gpu_uuids)
        gpu_metrics = _as_dict(compute_per_gpu.get(gpu_key))

        fp8_ratio, fp8_issues = _extract_fp8_ratio_for_gpu(gpu_metrics, gpu_uuid)
        bf16_ratio, bf16_issues = _extract_bf16_ratio_for_gpu(gpu_metrics, gpu_uuid)
        issues.extend(fp8_issues)
        issues.extend(bf16_issues)

        per_gpu_inputs.append(
            PerGPUScore(
                gpu_uuid=gpu_uuid,
                bw=bw_ratio,
                fp8=fp8_ratio,
                bf16=bf16_ratio,
                ar=ar_ratio,
            )
        )

    if not per_gpu_inputs:
        detail = _build_score_detail(
            ScoreStatus.ERROR,
            "No GPU data available for scoring",
            issues,
        )
        return ScoreResult(
            score_status=ScoreStatus.ERROR,
            score_status_detail=detail,
        )

    result = aggregate_gpu_scores(per_gpu_inputs)
    result.score_status_detail = _build_score_detail(
        result.score_status,
        result.score_status_detail,
        issues,
    )
    return result


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
