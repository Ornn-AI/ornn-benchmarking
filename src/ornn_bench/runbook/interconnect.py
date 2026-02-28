"""Interconnect benchmark matrix runner (Section 8.3).

Runs NCCL tests for all-reduce sweep, 1GB all-reduce variance,
reduce-scatter, all-gather, broadcast, and send-receive.
Captures bus bandwidth outputs.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runbook.parsers import parse_nccl_output
from ornn_bench.runner import SectionRunner

#: Required NCCL test matrix per Section 8.3
REQUIRED_NCCL_TESTS: tuple[str, ...] = (
    "all_reduce_sweep",
    "all_reduce_1gb",
    "reduce_scatter",
    "all_gather",
    "broadcast",
    "sendrecv",
)

#: Mapping from test names to binaries and arguments
NCCL_TEST_CONFIG: dict[str, dict[str, Any]] = {
    "all_reduce_sweep": {
        "binary": "all_reduce_perf",
        "args": ["-b", "8", "-e", "1G", "-f", "2", "-g", "2"],
    },
    "all_reduce_1gb": {
        "binary": "all_reduce_perf",
        "args": ["-b", "1G", "-e", "1G", "-n", "100", "-g", "2"],
    },
    "reduce_scatter": {
        "binary": "reduce_scatter_perf",
        "args": ["-b", "8", "-e", "1G", "-f", "2", "-g", "2"],
    },
    "all_gather": {
        "binary": "all_gather_perf",
        "args": ["-b", "8", "-e", "1G", "-f", "2", "-g", "2"],
    },
    "broadcast": {
        "binary": "broadcast_perf",
        "args": ["-b", "8", "-e", "1G", "-f", "2", "-g", "2"],
    },
    "sendrecv": {
        "binary": "sendrecv_perf",
        "args": ["-b", "8", "-e", "1G", "-f", "2", "-g", "2"],
    },
}


def _run_cmd(
    cmd: list[str],
    *,
    timeout: int = 600,
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


def run_nccl_test(
    test_name: str,
    *,
    raw_output: str | None = None,
) -> dict[str, Any]:
    """Run a single NCCL test.

    Parameters
    ----------
    test_name:
        Test name (e.g. "all_reduce_sweep", "sendrecv").
    raw_output:
        Pre-captured text output for testing.

    Returns
    -------
    dict with parsed NCCL results including bus bandwidth.
    """
    config = NCCL_TEST_CONFIG.get(test_name)
    if config is None:
        return {
            "test_key": test_name,
            "error": f"Unknown NCCL test: {test_name}",
            "devices": [],
            "results": [],
            "avg_bus_bandwidth": 0.0,
            "max_busbw": 0.0,
        }

    if raw_output is not None:
        parsed = parse_nccl_output(raw_output)
        parsed["test_key"] = test_name
        return parsed

    binary = config["binary"]
    args = config["args"]

    rc, stdout, stderr = _run_cmd([binary, *args])

    if rc != 0:
        return {
            "test_key": test_name,
            "binary": binary,
            "error": stderr or f"{binary} exited with code {rc}",
            "devices": [],
            "results": [],
            "avg_bus_bandwidth": 0.0,
            "max_busbw": 0.0,
        }

    parsed = parse_nccl_output(stdout)
    parsed["test_key"] = test_name
    return parsed


def collect_interconnect_matrix(
    *,
    test_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute the full interconnect benchmark matrix.

    Parameters
    ----------
    test_outputs:
        Pre-captured NCCL test outputs keyed by test name.

    Returns
    -------
    dict with results for all required NCCL tests.
    """
    results: dict[str, Any] = {
        "tests_run": list(REQUIRED_NCCL_TESTS),
        "nccl_results": {},
        "bus_bandwidth_summary": {},
    }

    for test_name in REQUIRED_NCCL_TESTS:
        raw = None
        if test_outputs and test_name in test_outputs:
            raw = test_outputs[test_name]

        test_result = run_nccl_test(test_name, raw_output=raw)
        results["nccl_results"][test_name] = test_result

        # Collect bus bandwidth summary
        results["bus_bandwidth_summary"][test_name] = {
            "avg_busbw": test_result.get("avg_bus_bandwidth", 0.0),
            "max_busbw": test_result.get("max_busbw", 0.0),
        }

    return results


class InterconnectMatrixRunner(SectionRunner):
    """Section runner for interconnect benchmarks (Section 8.3).

    Parameters
    ----------
    test_outputs:
        Pre-captured NCCL test outputs for testing.
    """

    def __init__(
        self,
        *,
        test_outputs: dict[str, str] | None = None,
    ) -> None:
        super().__init__("interconnect")
        self._test_outputs = test_outputs

    def run(self) -> SectionResult:
        """Execute interconnect benchmark matrix."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            metrics = collect_interconnect_matrix(
                test_outputs=self._test_outputs,
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
                error=f"Interconnect matrix failed: {exc}",
            )
