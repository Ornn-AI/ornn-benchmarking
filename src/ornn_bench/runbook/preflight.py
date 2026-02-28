"""Pre-flight inventory capture runner (Section 8.0).

Captures GPU UUIDs, NVLink topology/status, driver/CUDA/software versions,
OS/kernel, CPU/NUMA topology, and baseline NVLink/ECC/XID-related diagnostics.
"""

from __future__ import annotations

import platform
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


def _get_cpu_model() -> str:
    """Attempt to get CPU model string."""
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        rc, stdout, _ = _run_cmd(["sysctl", "-n", "machdep.cpu.brand_string"])
        if rc == 0 and stdout:
            return stdout.strip()
    return platform.processor() or "unknown"


def _get_numa_nodes() -> int:
    """Detect number of NUMA nodes on Linux."""
    try:
        rc, stdout, _ = _run_cmd(["lscpu"])
        if rc == 0:
            for line in stdout.splitlines():
                if "NUMA node(s)" in line:
                    return int(line.split(":")[-1].strip())
    except (ValueError, IndexError):
        pass
    return 0


def _get_pytorch_version() -> str:
    """Try to import torch and return version."""
    try:
        import torch  # type: ignore[import-not-found]
        return str(torch.__version__)
    except ImportError:
        return ""


def collect_preflight_inventory(
    nvidia_smi_q_output: str | None = None,
) -> dict[str, Any]:
    """Collect full pre-flight system inventory.

    Parameters
    ----------
    nvidia_smi_q_output:
        Pre-captured nvidia-smi -q output for testing. When None,
        runs nvidia-smi -q subprocess.

    Returns
    -------
    dict containing all inventory fields required by VAL-RUNBOOK-001.
    """
    inventory: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "kernel": platform.release(),
        "cpu_model": _get_cpu_model(),
        "numa_nodes": _get_numa_nodes(),
        "pytorch_version": _get_pytorch_version(),
        "gpu_inventory": {},
        "nvlink_topology": [],
        "ecc_baseline": {},
        "driver_version": "",
        "cuda_version": "",
        "software_versions": {
            "python": platform.python_version(),
        },
    }

    # Capture nvidia-smi -q
    raw_nvidia_smi_q = nvidia_smi_q_output
    if raw_nvidia_smi_q is None:
        rc, stdout, stderr = _run_cmd(["nvidia-smi", "-q"])
        if rc == 0:
            raw_nvidia_smi_q = stdout
        else:
            inventory["gpu_inventory"] = {"error": stderr or "nvidia-smi -q failed"}
            return inventory

    parsed = parse_nvidia_smi_q(raw_nvidia_smi_q)

    inventory["driver_version"] = parsed["driver_version"]
    inventory["cuda_version"] = parsed["cuda_version"]
    inventory["software_versions"]["driver"] = parsed["driver_version"]
    inventory["software_versions"]["cuda"] = parsed["cuda_version"]

    # Per-GPU details
    gpu_uuids: list[str] = []
    nvlink_links: list[dict[str, Any]] = []
    ecc_baseline: dict[str, Any] = {}

    for gpu in parsed["gpus"]:
        gpu_uuids.append(gpu["uuid"])
        nvlink_links.extend(gpu["nvlink"])

        # ECC baseline
        if gpu["ecc_errors"]:
            ecc_baseline[gpu["uuid"]] = {
                "ecc_mode": gpu["ecc_mode"],
                "errors": gpu["ecc_errors"],
            }

    inventory["gpu_inventory"] = {
        "attached_gpus": parsed["attached_gpus"],
        "gpus": parsed["gpus"],
        "gpu_uuids": gpu_uuids,
    }
    inventory["nvlink_topology"] = nvlink_links
    inventory["ecc_baseline"] = ecc_baseline

    return inventory


class PreflightRunner(SectionRunner):
    """Section runner for pre-flight inventory capture.

    Parameters
    ----------
    nvidia_smi_q_output:
        Pre-captured nvidia-smi -q output for testing/mocking.
    """

    def __init__(self, nvidia_smi_q_output: str | None = None) -> None:
        super().__init__("pre-flight")
        self._nvidia_smi_q_output = nvidia_smi_q_output

    def run(self) -> SectionResult:
        """Execute pre-flight inventory capture."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            inventory = collect_preflight_inventory(
                nvidia_smi_q_output=self._nvidia_smi_q_output,
            )

            finished_at = datetime.now(timezone.utc).isoformat()
            return SectionResult(
                name=self.name,
                status=BenchmarkStatus.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                metrics=inventory,
            )
        except Exception as exc:
            finished_at = datetime.now(timezone.utc).isoformat()
            return SectionResult(
                name=self.name,
                status=BenchmarkStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                error=f"Pre-flight inventory failed: {exc}",
            )
