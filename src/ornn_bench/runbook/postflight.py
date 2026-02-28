"""Post-flight consistency check runner (Section 8.5).

Validates UUID consistency between pre-flight and post-run states,
checks NVLink link status, and compares ECC error counters for
new errors introduced during the benchmark run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runbook.parsers import parse_nvidia_smi_q
from ornn_bench.runner import SectionRunner

# ---------------------------------------------------------------------------
# UUID consistency check
# ---------------------------------------------------------------------------


def check_uuid_consistency(
    pre_uuids: list[str],
    post_uuids: list[str],
) -> dict[str, Any]:
    """Check that GPU UUIDs are consistent between pre-flight and post-run.

    Parameters
    ----------
    pre_uuids:
        GPU UUIDs collected during pre-flight.
    post_uuids:
        GPU UUIDs collected after benchmark run.

    Returns
    -------
    dict with passed flag, missing_uuids, and new_uuids.
    """
    pre_set = set(pre_uuids)
    post_set = set(post_uuids)

    missing = sorted(pre_set - post_set)
    new = sorted(post_set - pre_set)

    return {
        "passed": len(missing) == 0 and len(new) == 0,
        "pre_count": len(pre_uuids),
        "post_count": len(post_uuids),
        "missing_uuids": missing,
        "new_uuids": new,
    }


# ---------------------------------------------------------------------------
# NVLink status check
# ---------------------------------------------------------------------------


def check_nvlink_status(
    post_nvidia_smi_q: str,
) -> dict[str, Any]:
    """Check NVLink link states from post-run nvidia-smi -q output.

    Parameters
    ----------
    post_nvidia_smi_q:
        Raw nvidia-smi -q output from post-run.

    Returns
    -------
    dict with passed flag, total_links, active_links, and inactive_links.
    """
    parsed = parse_nvidia_smi_q(post_nvidia_smi_q)

    total_links = 0
    active_links = 0
    inactive_links: list[dict[str, str]] = []

    for gpu in parsed["gpus"]:
        gpu_uuid = gpu.get("uuid", "unknown")
        for link in gpu.get("nvlink", []):
            total_links += 1
            state = link.get("state", "Unknown")
            if state == "Active":
                active_links += 1
            else:
                inactive_links.append({
                    "gpu_uuid": gpu_uuid,
                    "link_id": link.get("link_id", "?"),
                    "state": state,
                })

    return {
        "passed": len(inactive_links) == 0,
        "total_links": total_links,
        "active_links": active_links,
        "inactive_links": inactive_links,
    }


# ---------------------------------------------------------------------------
# ECC error check
# ---------------------------------------------------------------------------


def check_ecc_errors(
    pre_nvidia_smi_q: str,
    post_nvidia_smi_q: str,
) -> dict[str, Any]:
    """Compare ECC error counters between pre-flight and post-run.

    Detects any new volatile ECC errors introduced during the benchmark run.

    Parameters
    ----------
    pre_nvidia_smi_q:
        Pre-flight nvidia-smi -q output.
    post_nvidia_smi_q:
        Post-run nvidia-smi -q output.

    Returns
    -------
    dict with passed flag and new_errors keyed by GPU UUID.
    """
    pre_parsed = parse_nvidia_smi_q(pre_nvidia_smi_q)
    post_parsed = parse_nvidia_smi_q(post_nvidia_smi_q)

    # Index pre errors by GPU UUID
    pre_errors: dict[str, dict[str, int]] = {}
    for gpu in pre_parsed["gpus"]:
        pre_errors[gpu["uuid"]] = gpu.get("ecc_errors", {})

    new_errors: dict[str, dict[str, int]] = {}
    for gpu in post_parsed["gpus"]:
        uuid = gpu["uuid"]
        post_ecc = gpu.get("ecc_errors", {})
        pre_ecc = pre_errors.get(uuid, {})

        delta: dict[str, int] = {}
        for key, post_val in post_ecc.items():
            pre_val = pre_ecc.get(key, 0)
            if post_val > pre_val:
                delta[key] = post_val - pre_val

        if delta:
            new_errors[uuid] = delta

    return {
        "passed": len(new_errors) == 0,
        "new_errors": new_errors,
    }


# ---------------------------------------------------------------------------
# Full post-flight checks
# ---------------------------------------------------------------------------


def collect_postflight_checks(
    *,
    pre_nvidia_smi_q: str,
    post_nvidia_smi_q: str,
) -> dict[str, Any]:
    """Execute all post-flight consistency checks.

    Parameters
    ----------
    pre_nvidia_smi_q:
        Pre-flight nvidia-smi -q output.
    post_nvidia_smi_q:
        Post-run nvidia-smi -q output.

    Returns
    -------
    dict with results for each check and overall pass/fail.
    """
    pre_parsed = parse_nvidia_smi_q(pre_nvidia_smi_q)
    post_parsed = parse_nvidia_smi_q(post_nvidia_smi_q)

    pre_uuids = [gpu["uuid"] for gpu in pre_parsed["gpus"]]
    post_uuids = [gpu["uuid"] for gpu in post_parsed["gpus"]]

    uuid_result = check_uuid_consistency(pre_uuids, post_uuids)
    nvlink_result = check_nvlink_status(post_nvidia_smi_q)
    ecc_result = check_ecc_errors(pre_nvidia_smi_q, post_nvidia_smi_q)

    overall_passed = (
        uuid_result["passed"]
        and nvlink_result["passed"]
        and ecc_result["passed"]
    )

    return {
        "uuid_consistency": uuid_result,
        "nvlink_status": nvlink_result,
        "ecc_errors": ecc_result,
        "overall_passed": overall_passed,
    }


# ---------------------------------------------------------------------------
# PostflightRunner
# ---------------------------------------------------------------------------


class PostflightRunner(SectionRunner):
    """Section runner for post-flight consistency checks (Section 8.5).

    Parameters
    ----------
    pre_nvidia_smi_q:
        Pre-flight nvidia-smi -q output.
    post_nvidia_smi_q:
        Post-run nvidia-smi -q output. When None, runs subprocess.
    """

    def __init__(
        self,
        *,
        pre_nvidia_smi_q: str,
        post_nvidia_smi_q: str | None = None,
    ) -> None:
        super().__init__("post-flight")
        self._pre_nvidia_smi_q = pre_nvidia_smi_q
        self._post_nvidia_smi_q = post_nvidia_smi_q

    def run(self) -> SectionResult:
        """Execute post-flight consistency checks."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            # If post-run output not provided, capture it
            post_raw = self._post_nvidia_smi_q
            if post_raw is None:
                import subprocess
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "-q"],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode == 0:
                        post_raw = result.stdout
                    else:
                        finished_at = datetime.now(timezone.utc).isoformat()
                        return SectionResult(
                            name=self.name,
                            status=BenchmarkStatus.FAILED,
                            started_at=started_at,
                            finished_at=finished_at,
                            error="Failed to capture post-run nvidia-smi -q",
                        )
                except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                    finished_at = datetime.now(timezone.utc).isoformat()
                    return SectionResult(
                        name=self.name,
                        status=BenchmarkStatus.FAILED,
                        started_at=started_at,
                        finished_at=finished_at,
                        error=f"nvidia-smi not available: {exc}",
                    )

            metrics = collect_postflight_checks(
                pre_nvidia_smi_q=self._pre_nvidia_smi_q,
                post_nvidia_smi_q=post_raw,
            )

            finished_at = datetime.now(timezone.utc).isoformat()
            return SectionResult(
                name=self.name,
                status=BenchmarkStatus.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                metrics=metrics,
            )
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            return SectionResult(
                name=self.name,
                status=BenchmarkStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                error=f"Post-flight checks failed: {exc}",
            )
