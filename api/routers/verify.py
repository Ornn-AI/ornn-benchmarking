"""Router for benchmark score verification.

Endpoint:
  - ``POST /api/v1/verify`` — verify submitted scores against server recomputation
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import require_api_key
from api.dependencies import get_rate_limiter
from api.models import MetricDetailResponse, VerifyRequest, VerifyResponse
from api.rate_limit import RateLimiter
from api.scoring import recompute_and_verify

router = APIRouter(prefix="/api/v1", tags=["verify"])

_require_api_key = Depends(require_api_key)
_get_rate_limiter = Depends(get_rate_limiter)


@router.post("/verify", response_model=VerifyResponse)
async def verify_scores(
    payload: VerifyRequest,
    api_key: str = _require_api_key,
    limiter: RateLimiter = _get_rate_limiter,
) -> VerifyResponse:
    """Verify submitted scores against server-side recomputation.

    Accepts raw metric components and client-computed scores, recomputes
    Ornn-I and Ornn-T on the server, and returns ``verified`` if all scores
    match within tolerance or ``mismatch`` with per-metric details showing
    exactly which metrics diverged and by how much.

    Requires a valid API key.  Subject to rate limiting.
    """
    # --- Rate limiting --------------------------------------------------
    allowed, retry_after = limiter.check(api_key)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please retry later.",
            headers={"Retry-After": str(retry_after)},
        )

    # --- Recompute and verify -------------------------------------------
    result = recompute_and_verify(
        components=payload.components,
        submitted_ornn_i=payload.ornn_i,
        submitted_ornn_t=payload.ornn_t,
        submitted_qualification=payload.qualification,
    )

    # Convert internal result to response model
    metric_details = [
        MetricDetailResponse(
            metric=d.metric,
            submitted=d.submitted,
            server_computed=d.server_computed,
            match=d.match,
            delta=d.delta,
        )
        for d in result.metric_details
    ]

    return VerifyResponse(
        status=result.status.value,
        server_ornn_i=result.server_ornn_i,
        server_ornn_t=result.server_ornn_t,
        server_qualification=result.server_qualification,
        metric_details=metric_details,
        tolerance=result.tolerance,
    )
