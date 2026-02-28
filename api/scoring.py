"""Server-side scoring and qualification engine for the Ornn Benchmarking API.

Replicates the deterministic scoring formulas from the CLI scoring module
(``ornn_bench.scoring``) to enable server-side verification of submitted
scores.  Uses the same reference values, formulas, and qualification gates
so that local and server-computed scores are always consistent.

Endpoints use :func:`recompute_and_verify` to compare submitted scores
against server recomputation and return explicit verified/mismatch results
with per-metric details.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Reference values — must match the CLI scoring module exactly
# ---------------------------------------------------------------------------

DEFAULT_REFS: dict[str, float] = {
    "bw_ref": 1.0,
    "fp8_ref": 1.0,
    "bf16_ref": 1.0,
    "ar_ref": 1.0,
}

# Default tolerance for score comparison (absolute delta)
DEFAULT_TOLERANCE: float = 0.01


# ---------------------------------------------------------------------------
# Qualification enum (mirrors ornn_bench.models.Qualification)
# ---------------------------------------------------------------------------


class Qualification(str, Enum):
    """GPU qualification grade."""

    PREMIUM = "Premium"
    STANDARD = "Standard"
    BELOW = "Below"


# ---------------------------------------------------------------------------
# Metric validation helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Score computation (deterministic formulas)
# ---------------------------------------------------------------------------


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
    assert bw is not None and fp8 is not None  # type narrowing
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
    assert bf16 is not None and ar is not None  # type narrowing
    return 55.0 * (bf16 / bf16_ref) + 45.0 * (ar / ar_ref)


def determine_qualification(
    ornn_i: float | None,
    ornn_t: float | None,
) -> str | None:
    """Determine qualification grade based on scores.

    Applies floor + composite gate logic:
      - Premium:  composite >= 90 AND both floors >= 80
      - Standard: composite >= 70 AND both floors >= 60
      - Below:    everything else

    Returns None if either score is None.
    """
    if ornn_i is None or ornn_t is None:
        return None
    composite = (ornn_i + ornn_t) / 2.0
    if composite >= 90.0 and ornn_i >= 80.0 and ornn_t >= 80.0:
        return Qualification.PREMIUM.value
    if composite >= 70.0 and ornn_i >= 60.0 and ornn_t >= 60.0:
        return Qualification.STANDARD.value
    return Qualification.BELOW.value


# ---------------------------------------------------------------------------
# Verification result types
# ---------------------------------------------------------------------------


class VerificationStatus(str, Enum):
    """Outcome of a score verification."""

    VERIFIED = "verified"
    MISMATCH = "mismatch"


@dataclass
class MetricDetail:
    """Per-metric comparison detail."""

    metric: str
    submitted: float | None
    server_computed: float | None
    match: bool
    delta: float | None = None


@dataclass
class VerificationResult:
    """Full verification outcome with per-metric details."""

    status: VerificationStatus
    server_ornn_i: float | None
    server_ornn_t: float | None
    server_qualification: str | None
    metric_details: list[MetricDetail] = field(default_factory=list)
    tolerance: float = DEFAULT_TOLERANCE


# ---------------------------------------------------------------------------
# Recompute & verify
# ---------------------------------------------------------------------------


def _compare_score(
    metric_name: str,
    submitted: float | None,
    computed: float | None,
    tolerance: float,
) -> MetricDetail:
    """Compare a single submitted score value against server computation."""
    if submitted is None and computed is None:
        return MetricDetail(
            metric=metric_name,
            submitted=submitted,
            server_computed=computed,
            match=True,
            delta=None,
        )
    if submitted is None or computed is None:
        return MetricDetail(
            metric=metric_name,
            submitted=submitted,
            server_computed=computed,
            match=False,
            delta=None,
        )
    delta = abs(submitted - computed)
    return MetricDetail(
        metric=metric_name,
        submitted=submitted,
        server_computed=computed,
        match=delta <= tolerance,
        delta=delta,
    )


def recompute_and_verify(
    components: dict[str, float],
    submitted_ornn_i: float | None,
    submitted_ornn_t: float | None,
    submitted_qualification: str | None,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    ref_overrides: dict[str, float] | None = None,
) -> VerificationResult:
    """Recompute scores from raw components and verify against submitted values.

    Parameters
    ----------
    components:
        Raw metric components: ``bw``, ``fp8``, ``bf16``, ``ar``.
    submitted_ornn_i:
        The Ornn-I score submitted by the client.
    submitted_ornn_t:
        The Ornn-T score submitted by the client.
    submitted_qualification:
        The qualification grade submitted by the client.
    tolerance:
        Absolute tolerance for score comparison (default 0.01).
    ref_overrides:
        Optional reference value overrides (keys: bw_ref, fp8_ref, bf16_ref, ar_ref).

    Returns
    -------
    VerificationResult with ``verified`` or ``mismatch`` status and per-metric
    detail.
    """
    refs = {**DEFAULT_REFS}
    if ref_overrides:
        refs.update(ref_overrides)

    # Extract component values
    bw = components.get("bw")
    fp8 = components.get("fp8")
    bf16 = components.get("bf16")
    ar = components.get("ar")

    # Recompute scores
    server_ornn_i = compute_ornn_i(
        bw, fp8, bw_ref=refs["bw_ref"], fp8_ref=refs["fp8_ref"]
    )
    server_ornn_t = compute_ornn_t(
        bf16, ar, bf16_ref=refs["bf16_ref"], ar_ref=refs["ar_ref"]
    )
    server_qualification = determine_qualification(server_ornn_i, server_ornn_t)

    # Build per-metric details
    details: list[MetricDetail] = [
        _compare_score("ornn_i", submitted_ornn_i, server_ornn_i, tolerance),
        _compare_score("ornn_t", submitted_ornn_t, server_ornn_t, tolerance),
    ]

    # Check qualification match (string comparison, case-sensitive)
    qual_match = submitted_qualification == server_qualification
    details.append(
        MetricDetail(
            metric="qualification",
            submitted=None,
            server_computed=None,
            match=qual_match,
            delta=None,
        )
    )

    # Overall status: verified only if ALL metrics match
    all_match = all(d.match for d in details)
    status = VerificationStatus.VERIFIED if all_match else VerificationStatus.MISMATCH

    return VerificationResult(
        status=status,
        server_ornn_i=server_ornn_i,
        server_ornn_t=server_ornn_t,
        server_qualification=server_qualification,
        metric_details=details,
        tolerance=tolerance,
    )
