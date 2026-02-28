"""Memory benchmark matrix runner (Section 8.2).

Runs nvbandwidth for all required test types plus PyTorch D2D cross-validation.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runbook.parsers import parse_nvbandwidth_json
from ornn_bench.runner import SectionRunner

#: Required nvbandwidth test types per Section 8.2
REQUIRED_TESTS: tuple[str, ...] = (
    "device_local_copy",
    "device_local_copy_sm",
    "h2d",
    "d2h",
    "d2d_read",
    "d2d_write",
    "d2d_bidir",
)

#: Mapping from short names to nvbandwidth test names
NVBW_TEST_MAP: dict[str, str] = {
    "device_local_copy": "device_to_device_memcpy_read_ce",
    "device_local_copy_sm": "device_to_device_memcpy_read_sm",
    "h2d": "host_to_device_memcpy_ce",
    "d2h": "device_to_host_memcpy_ce",
    "d2d_read": "device_to_device_memcpy_read_ce",
    "d2d_write": "device_to_device_memcpy_write_ce",
    "d2d_bidir": "device_to_device_bidirectional_memcpy_read_ce",
}


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a subprocess with optional environment."""
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", f"Command timed out after {timeout}s"
    except OSError as exc:
        return -1, "", f"OS error: {exc}"


def run_nvbandwidth_test(
    test_name: str,
    *,
    raw_output: str | None = None,
) -> dict[str, Any]:
    """Run a single nvbandwidth test.

    Parameters
    ----------
    test_name:
        Short test name (e.g. "h2d", "d2d_read").
    raw_output:
        Pre-captured JSON output for testing.

    Returns
    -------
    dict with parsed bandwidth results.
    """
    if raw_output is not None:
        parsed = parse_nvbandwidth_json(raw_output)
        parsed["test_key"] = test_name
        return parsed

    nvbw_test = NVBW_TEST_MAP.get(test_name, test_name)
    rc, stdout, stderr = _run_cmd(
        ["nvbandwidth", "-t", nvbw_test, "-j"],
    )

    if rc != 0:
        return {
            "test_key": test_name,
            "testname": nvbw_test,
            "error": stderr or f"nvbandwidth exited with code {rc}",
            "bandwidth_matrix": [],
            "sum": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    parsed = parse_nvbandwidth_json(stdout)
    parsed["test_key"] = test_name
    return parsed


def run_pytorch_d2d_crossval(
    *,
    raw_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run PyTorch D2D bandwidth cross-validation.

    Parameters
    ----------
    raw_result:
        Pre-computed result for testing.

    Returns
    -------
    dict with cross-validation bandwidth measurement.
    """
    if raw_result is not None:
        return raw_result

    # Attempt to import torch and run cross-validation
    try:
        import torch  # type: ignore[import-not-found]

        if not torch.cuda.is_available():
            return {
                "test_key": "pytorch_d2d_crossval",
                "error": "CUDA not available in PyTorch",
                "bandwidth_gb_s": 0.0,
            }

        gpu_count = torch.cuda.device_count()
        if gpu_count < 1:
            return {
                "test_key": "pytorch_d2d_crossval",
                "error": "No CUDA devices found",
                "bandwidth_gb_s": 0.0,
            }

        # Simple D2D bandwidth measurement
        size = 256 * 1024 * 1024  # 256MB
        src = torch.randn(size // 4, device="cuda:0")

        if gpu_count > 1:
            dst = torch.empty_like(src, device="cuda:1")
        else:
            dst = torch.empty_like(src, device="cuda:0")

        # Warmup
        for _ in range(5):
            dst.copy_(src)

        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(20):
            dst.copy_(src)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event) / 20.0
        bandwidth_gb_s = (size / 1e9) / (elapsed_ms / 1e3)

        return {
            "test_key": "pytorch_d2d_crossval",
            "bandwidth_gb_s": round(bandwidth_gb_s, 2),
            "size_bytes": size,
            "iterations": 20,
        }

    except ImportError:
        return {
            "test_key": "pytorch_d2d_crossval",
            "error": "PyTorch not available",
            "bandwidth_gb_s": 0.0,
        }
    except Exception as exc:
        return {
            "test_key": "pytorch_d2d_crossval",
            "error": f"PyTorch D2D cross-validation failed: {exc}",
            "bandwidth_gb_s": 0.0,
        }


def collect_memory_matrix(
    *,
    test_outputs: dict[str, str] | None = None,
    pytorch_d2d_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the full memory benchmark matrix.

    Parameters
    ----------
    test_outputs:
        Pre-captured nvbandwidth JSON outputs keyed by test name.
    pytorch_d2d_result:
        Pre-computed PyTorch D2D result for testing.

    Returns
    -------
    dict with results for all required tests plus cross-validation.
    """
    results: dict[str, Any] = {
        "tests_run": list(REQUIRED_TESTS),
        "nvbandwidth_results": {},
        "pytorch_d2d_crossval": {},
    }

    for test_name in REQUIRED_TESTS:
        raw = None
        if test_outputs and test_name in test_outputs:
            raw = test_outputs[test_name]

        test_result = run_nvbandwidth_test(test_name, raw_output=raw)
        results["nvbandwidth_results"][test_name] = test_result

    # PyTorch cross-validation
    results["pytorch_d2d_crossval"] = run_pytorch_d2d_crossval(
        raw_result=pytorch_d2d_result,
    )

    return results


class MemoryMatrixRunner(SectionRunner):
    """Section runner for memory benchmarks (Section 8.2).

    Parameters
    ----------
    test_outputs:
        Pre-captured nvbandwidth outputs for testing.
    pytorch_d2d_result:
        Pre-computed PyTorch D2D result for testing.
    """

    def __init__(
        self,
        *,
        test_outputs: dict[str, str] | None = None,
        pytorch_d2d_result: dict[str, Any] | None = None,
    ) -> None:
        super().__init__("memory")
        self._test_outputs = test_outputs
        self._pytorch_d2d_result = pytorch_d2d_result

    def run(self) -> SectionResult:
        """Execute memory benchmark matrix."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            metrics = collect_memory_matrix(
                test_outputs=self._test_outputs,
                pytorch_d2d_result=self._pytorch_d2d_result,
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
                error=f"Memory matrix failed: {exc}",
            )
