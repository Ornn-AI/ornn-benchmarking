"""HTTP client for the Ornn Benchmarking API.

Provides upload, verification, and schema-version checking against the
remote API, with retry-safe semantics backed by the API's idempotent
dedupe contract.

All network errors surface as :class:`UploadError` subclasses so that
CLI callers can display clear messages without exposing raw tracebacks.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import httpx

from ornn_bench.models import BenchmarkReport

# ---------------------------------------------------------------------------
# Default API base URL (Cloud Run deployment)
# ---------------------------------------------------------------------------

DEFAULT_API_URL = "https://ornn-benchmarking-api.run.app"

# Supported schema versions that the API accepts.
# The CLI checks this locally before attempting upload.
SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}

# Timeout settings for HTTP requests (seconds)
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class UploadError(Exception):
    """Base class for upload-related errors."""


class AuthenticationError(UploadError):
    """API key is missing, invalid, or revoked."""


class SchemaVersionError(UploadError):
    """Report uses an unsupported schema version."""


class ValidationError(UploadError):
    """API rejected the payload due to validation errors."""


class RateLimitError(UploadError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ServerError(UploadError):
    """Unexpected server-side error."""


class NetworkError(UploadError):
    """Network connectivity failure."""


# ---------------------------------------------------------------------------
# Upload response
# ---------------------------------------------------------------------------


@dataclass
class UploadResult:
    """Successful upload result from the API."""

    run_id: str
    received_at: str
    stored_at: str
    is_duplicate: bool = False


# ---------------------------------------------------------------------------
# Verification response
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """Per-metric comparison from server verification."""

    metric: str
    submitted: float | None
    server_computed: float | None
    match: bool
    delta: float | None = None


@dataclass
class VerifyResult:
    """Verification result from the API."""

    status: str  # "verified" or "mismatch"
    server_ornn_i: float | None
    server_ornn_t: float | None
    server_qualification: str | None
    metric_details: list[MetricComparison] = field(default_factory=list)
    tolerance: float = 0.01


# ---------------------------------------------------------------------------
# Dedupe key computation (mirrors API-side logic)
# ---------------------------------------------------------------------------


def compute_dedupe_key(report: BenchmarkReport) -> str:
    """Compute the deterministic dedupe key for a report.

    Must match the server-side ``_compute_dedupe_key`` in
    ``api.routers.runs`` so that retries are idempotent.
    """
    identity = json.dumps(
        {
            "report_id": report.report_id,
            "created_at": report.created_at,
            "schema_version": report.schema_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Report validation (local, before network call)
# ---------------------------------------------------------------------------


def validate_report_for_upload(report: BenchmarkReport) -> list[str]:
    """Validate a report locally before attempting upload.

    Returns a list of validation error messages (empty if valid).
    Checks required fields and schema version compatibility.
    """
    errors: list[str] = []

    if not report.report_id:
        errors.append("Missing report_id")
    if not report.created_at:
        errors.append("Missing created_at timestamp")
    if not report.schema_version:
        errors.append("Missing schema_version")
    elif report.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        errors.append(
            f"Unsupported schema version '{report.schema_version}'. "
            f"Supported versions: {', '.join(sorted(SUPPORTED_SCHEMA_VERSIONS))}. "
            f"Please update ornn-bench to a compatible version."
        )

    return errors


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


class OrnnApiClient:
    """HTTP client for the Ornn Benchmarking API.

    Handles upload, verification, and error mapping. Retry-safe via
    the API's idempotent dedupe contract.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        api_key: str = "",
        timeout: tuple[float, float] = (CONNECT_TIMEOUT, READ_TIMEOUT),
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._timeout = httpx.Timeout(
            timeout[1], connect=timeout[0],
        )

    def _headers(self) -> dict[str, str]:
        """Build request headers with API key."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Map HTTP error codes to typed exceptions."""
        status_code = response.status_code

        try:
            body = response.json()
            detail = body.get("detail", response.text)
        except Exception:
            detail = response.text

        if status_code == 401:
            raise AuthenticationError(
                f"Authentication failed: {detail}"
            )
        elif status_code == 422:
            raise ValidationError(
                f"Validation error: {detail}"
            )
        elif status_code == 429:
            retry_after_header = response.headers.get("Retry-After")
            retry_after = int(retry_after_header) if retry_after_header else None
            raise RateLimitError(
                f"Rate limit exceeded. {detail}",
                retry_after=retry_after,
            )
        elif status_code >= 500:
            raise ServerError(
                f"Server error ({status_code}): {detail}"
            )
        else:
            raise UploadError(
                f"Unexpected error ({status_code}): {detail}"
            )

    def upload(self, report: BenchmarkReport) -> UploadResult:
        """Upload a benchmark report to the API.

        Performs local validation first, then sends the report.
        Returns an :class:`UploadResult` with the server-assigned run_id.

        The API's idempotent dedupe contract means retrying the same
        report after a transient failure is safe — duplicates are
        detected and the existing run_id is returned.

        Raises
        ------
        SchemaVersionError
            If the report's schema version is not supported.
        AuthenticationError
            If the API key is invalid or missing.
        ValidationError
            If the API rejects the payload.
        RateLimitError
            If rate limit is exceeded.
        NetworkError
            If there is a connectivity failure.
        ServerError
            If the server returns a 5xx status.
        """
        # Local validation before network call
        validation_errors = validate_report_for_upload(report)
        if validation_errors:
            # Check for schema version error specifically
            for err in validation_errors:
                if "schema version" in err.lower():
                    raise SchemaVersionError(err)
            raise ValidationError(
                f"Report validation failed: {'; '.join(validation_errors)}"
            )

        url = f"{self.api_url}/api/v1/runs"
        payload = report.model_dump_json()

        try:
            response = httpx.post(
                url,
                content=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.ConnectError as exc:
            raise NetworkError(
                f"Failed to connect to API at {self.api_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise NetworkError(
                f"Request timed out connecting to {self.api_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"Network error: {exc}"
            ) from exc

        if response.status_code in (200, 201):
            data = response.json()
            return UploadResult(
                run_id=data["run_id"],
                received_at=data["received_at"],
                stored_at=data["stored_at"],
                is_duplicate=(response.status_code == 200),
            )

        self._handle_error_response(response)
        # Should never reach here, but satisfy type checker
        raise UploadError("Unexpected state")  # pragma: no cover

    def verify(self, report: BenchmarkReport) -> VerifyResult:
        """Verify local scores against server recomputation.

        Sends the report's component metrics and scores to the
        verification endpoint and returns per-metric comparison details.

        Raises the same error types as :meth:`upload`.
        """
        url = f"{self.api_url}/api/v1/verify"
        payload = {
            "components": report.scores.components,
            "ornn_i": report.scores.ornn_i,
            "ornn_t": report.scores.ornn_t,
            "qualification": (
                report.scores.qualification.value
                if report.scores.qualification
                else None
            ),
        }

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
        except httpx.ConnectError as exc:
            raise NetworkError(
                f"Failed to connect to API at {self.api_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise NetworkError(
                f"Request timed out connecting to {self.api_url}: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise NetworkError(
                f"Network error: {exc}"
            ) from exc

        if response.status_code == 200:
            data = response.json()
            metric_details = [
                MetricComparison(
                    metric=d["metric"],
                    submitted=d.get("submitted"),
                    server_computed=d.get("server_computed"),
                    match=d["match"],
                    delta=d.get("delta"),
                )
                for d in data.get("metric_details", [])
            ]
            return VerifyResult(
                status=data["status"],
                server_ornn_i=data.get("server_ornn_i"),
                server_ornn_t=data.get("server_ornn_t"),
                server_qualification=data.get("server_qualification"),
                metric_details=metric_details,
                tolerance=data.get("tolerance", 0.01),
            )

        self._handle_error_response(response)
        raise UploadError("Unexpected state")  # pragma: no cover
