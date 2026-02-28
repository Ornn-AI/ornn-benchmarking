"""Thermal/power monitoring runner (Section 8.4).

Captures continuous nvidia-smi dmon time-series during benchmark phases,
pre/post system state snapshots, and XID error detection output.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from typing import Any

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runbook.parsers import parse_nvidia_smi_q
from ornn_bench.runner import SectionRunner


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = 30,
) -> tuple[int, str, str]:
    """Run a subprocess and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except OSError as exc:
        return -1, "", f"OS error: {exc}"


# ---------------------------------------------------------------------------
# dmon parser
# ---------------------------------------------------------------------------


def parse_dmon_output(raw: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi dmon output into structured time-series entries.

    Expected columns: gpu pwr gtemp mtemp sm mem enc dec mclk pclk

    Parameters
    ----------
    raw:
        Raw text output from ``nvidia-smi dmon``.

    Returns
    -------
    list of dicts, one per sample row.
    """
    entries: list[dict[str, Any]] = []

    for line in raw.splitlines():
        stripped = line.strip()
        # Skip empty lines and comment/header lines
        if not stripped or stripped.startswith("#"):
            continue

        parts = stripped.split()
        if len(parts) < 10:
            continue

        try:
            entry: dict[str, Any] = {
                "gpu_index": int(parts[0]),
                "power_w": int(parts[1]),
                "gpu_temp_c": int(parts[2]),
                "mem_temp_c": int(parts[3]),
                "sm_util_pct": int(parts[4]),
                "mem_util_pct": int(parts[5]),
                "enc_util_pct": int(parts[6]),
                "dec_util_pct": int(parts[7]),
                "mem_clock_mhz": int(parts[8]),
                "gpu_clock_mhz": int(parts[9]),
            }
            entries.append(entry)
        except (ValueError, IndexError):
            continue

    return entries


# ---------------------------------------------------------------------------
# XID error parser
# ---------------------------------------------------------------------------

_XID_RE = re.compile(
    r"NVRM:\s*Xid\s*\(PCI:([^)]+)\):\s*(\d+)"
)


def parse_xid_errors(raw: str) -> list[dict[str, Any]]:
    """Parse XID error entries from dmesg or kernel log output.

    Parameters
    ----------
    raw:
        Raw text from dmesg/syslog containing potential Xid messages.

    Returns
    -------
    list of dicts with xid_code and pci_id for each XID error found.
    """
    errors: list[dict[str, Any]] = []

    for line in raw.splitlines():
        match = _XID_RE.search(line)
        if match:
            errors.append({
                "pci_id": match.group(1).strip(),
                "xid_code": int(match.group(2)),
                "raw_line": line.strip(),
            })

    return errors


# ---------------------------------------------------------------------------
# Monitoring data collection
# ---------------------------------------------------------------------------


def collect_monitoring_data(
    *,
    pre_snapshot_raw: str | None = None,
    post_snapshot_raw: str | None = None,
    dmon_raw: str | None = None,
    xid_raw: str | None = None,
) -> dict[str, Any]:
    """Collect full monitoring data for a benchmark run.

    Parameters
    ----------
    pre_snapshot_raw:
        Pre-benchmark nvidia-smi -q output. When None, runs subprocess.
    post_snapshot_raw:
        Post-benchmark nvidia-smi -q output. When None, runs subprocess.
    dmon_raw:
        Pre-captured dmon output. When None, expected to be collected
        during benchmark phases.
    xid_raw:
        Pre-captured XID/dmesg output. When None, runs subprocess.

    Returns
    -------
    dict containing pre/post snapshots, time-series, and XID errors.
    """
    result: dict[str, Any] = {
        "pre_snapshot": {},
        "post_snapshot": {},
        "time_series": [],
        "xid_errors": [],
    }

    # Pre-snapshot
    if pre_snapshot_raw is None:
        rc, stdout, stderr = _run_cmd(["nvidia-smi", "-q"])
        if rc == 0:
            pre_snapshot_raw = stdout
        else:
            result["pre_snapshot"] = {"error": stderr or "nvidia-smi -q failed"}
    if pre_snapshot_raw:
        result["pre_snapshot"] = parse_nvidia_smi_q(pre_snapshot_raw)

    # Post-snapshot
    if post_snapshot_raw is None:
        rc, stdout, stderr = _run_cmd(["nvidia-smi", "-q"])
        if rc == 0:
            post_snapshot_raw = stdout
        else:
            result["post_snapshot"] = {"error": stderr or "nvidia-smi -q failed"}
    if post_snapshot_raw:
        result["post_snapshot"] = parse_nvidia_smi_q(post_snapshot_raw)

    # Time-series (dmon)
    if dmon_raw is None:
        # In real usage, dmon runs in background during benchmarks.
        # When not provided, attempt a single snapshot.
        rc, stdout, stderr = _run_cmd(
            ["nvidia-smi", "dmon", "-c", "1"],
            timeout=10,
        )
        if rc == 0:
            dmon_raw = stdout
    if dmon_raw:
        result["time_series"] = parse_dmon_output(dmon_raw)

    # XID errors
    if xid_raw is None:
        rc, stdout, stderr = _run_cmd(["dmesg"], timeout=10)
        xid_raw = stdout if rc == 0 else ""
    result["xid_errors"] = parse_xid_errors(xid_raw)

    return result


# ---------------------------------------------------------------------------
# MonitoringRunner
# ---------------------------------------------------------------------------


class MonitoringRunner(SectionRunner):
    """Section runner for thermal/power monitoring (Section 8.4).

    Parameters
    ----------
    pre_snapshot_raw:
        Pre-captured nvidia-smi -q output.
    post_snapshot_raw:
        Post-captured nvidia-smi -q output.
    dmon_raw:
        Pre-captured dmon output.
    xid_raw:
        Pre-captured XID/dmesg output.
    """

    def __init__(
        self,
        *,
        pre_snapshot_raw: str | None = None,
        post_snapshot_raw: str | None = None,
        dmon_raw: str | None = None,
        xid_raw: str | None = None,
    ) -> None:
        super().__init__("monitoring")
        self._pre_snapshot_raw = pre_snapshot_raw
        self._post_snapshot_raw = post_snapshot_raw
        self._dmon_raw = dmon_raw
        self._xid_raw = xid_raw

    def run(self) -> SectionResult:
        """Execute monitoring data collection."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            metrics = collect_monitoring_data(
                pre_snapshot_raw=self._pre_snapshot_raw,
                post_snapshot_raw=self._post_snapshot_raw,
                dmon_raw=self._dmon_raw,
                xid_raw=self._xid_raw,
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
                error=f"Monitoring capture failed: {exc}",
            )
