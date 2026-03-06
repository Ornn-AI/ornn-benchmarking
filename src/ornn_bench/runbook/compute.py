"""Compute benchmark matrix runner (Section 8.1).

Runs MAMF (Matrix Multiply Performance Finder) for required dtypes
(BF16, FP8 E4M3, FP8 E5M2, FP16, optional TF32) plus fixed-shape checks.
Each GPU is isolated via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from ornn_bench.models import BenchmarkStatus, SectionResult
from ornn_bench.runbook.parsers import parse_mamf_output
from ornn_bench.runner import SectionRunner
from ornn_bench.system import detect_gpu_count

#: Required dtypes for compute matrix
REQUIRED_DTYPES: tuple[str, ...] = ("bf16", "fp8_e4m3", "fp8_e5m2", "fp16")

#: Optional dtypes
OPTIONAL_DTYPES: tuple[str, ...] = ("tf32",)

#: Default fixed shape for verification
DEFAULT_FIXED_SHAPE: dict[str, int] = {"m": 4096, "n": 4096, "k": 4096}


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


def run_mamf_for_dtype(
    dtype: str,
    gpu_index: int = 0,
    *,
    mamf_cmd: str = "mamf-finder.py",
    raw_output: str | None = None,
) -> dict[str, Any]:
    """Run mamf-finder for a single dtype on a specific GPU.

    Parameters
    ----------
    dtype:
        Data type to benchmark (e.g. "bf16", "fp8_e4m3").
    gpu_index:
        GPU index to isolate via CUDA_VISIBLE_DEVICES.
    mamf_cmd:
        Path to mamf-finder.py script.
    raw_output:
        Pre-captured output for testing. Skips subprocess when provided.

    Returns
    -------
    dict with parsed results plus metadata (gpu_index, dtype, raw_output ref).
    """
    if raw_output is not None:
        parsed = parse_mamf_output(raw_output)
        parsed["gpu_index"] = gpu_index
        parsed["cuda_visible_devices"] = str(gpu_index)
        return parsed

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_index)}
    rc, stdout, stderr = _run_cmd(
        ["python", mamf_cmd, "--dtype", dtype, "--m_range", "1024", "8192",
         "--n_range", "1024", "8192", "--k_range", "1024", "8192"],
        env=env,
    )

    if rc != 0:
        return {
            "dtype": dtype,
            "gpu_index": gpu_index,
            "error": stderr or f"mamf-finder exited with code {rc}",
            "cuda_visible_devices": str(gpu_index),
            "results": [],
            "best": None,
        }

    parsed = parse_mamf_output(stdout)
    parsed["gpu_index"] = gpu_index
    parsed["cuda_visible_devices"] = str(gpu_index)
    return parsed


def run_mamf_fixed_shape(
    dtype: str,
    gpu_index: int = 0,
    shape: dict[str, int] | None = None,
    *,
    mamf_cmd: str = "mamf-finder.py",
    raw_output: str | None = None,
) -> dict[str, Any]:
    """Run mamf-finder for a fixed shape.

    Parameters
    ----------
    dtype:
        Data type to benchmark.
    gpu_index:
        GPU index.
    shape:
        Dict with m, n, k dimensions. Defaults to 4096x4096x4096.
    mamf_cmd:
        Path to mamf-finder.py script.
    raw_output:
        Pre-captured output for testing.

    Returns
    -------
    dict with parsed fixed-shape results.
    """
    if shape is None:
        shape = dict(DEFAULT_FIXED_SHAPE)

    if raw_output is not None:
        parsed = parse_mamf_output(raw_output)
        parsed["gpu_index"] = gpu_index
        parsed["cuda_visible_devices"] = str(gpu_index)
        parsed["fixed_shape"] = shape
        return parsed

    env = {"CUDA_VISIBLE_DEVICES": str(gpu_index)}
    rc, stdout, stderr = _run_cmd(
        ["python", mamf_cmd, "--dtype", dtype,
         "--m", str(shape["m"]), "--n", str(shape["n"]), "--k", str(shape["k"])],
        env=env,
    )

    if rc != 0:
        return {
            "dtype": dtype,
            "gpu_index": gpu_index,
            "fixed_shape": shape,
            "error": stderr or f"mamf-finder exited with code {rc}",
            "cuda_visible_devices": str(gpu_index),
            "results": [],
            "best": None,
        }

    parsed = parse_mamf_output(stdout)
    parsed["gpu_index"] = gpu_index
    parsed["cuda_visible_devices"] = str(gpu_index)
    parsed["fixed_shape"] = shape
    return parsed


def collect_compute_matrix(
    gpu_count: int | None = None,
    *,
    include_tf32: bool = False,
    dtype_outputs: dict[str, str] | None = None,
    fixed_shape_outputs: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Execute the full compute benchmark matrix.

    Parameters
    ----------
    gpu_count:
        Number of GPUs to run on.
    include_tf32:
        Whether to include TF32 benchmarks.
    dtype_outputs:
        Pre-captured outputs keyed by dtype for testing.
    fixed_shape_outputs:
        Pre-captured fixed-shape outputs keyed by dtype for testing.

    Returns
    -------
    dict with per-GPU, per-dtype results and fixed-shape results.
    """
    resolved_gpu_count = gpu_count if gpu_count is not None else detect_gpu_count()

    dtypes = list(REQUIRED_DTYPES)
    if include_tf32:
        dtypes.append("tf32")

    results: dict[str, Any] = {
        "gpu_count": resolved_gpu_count,
        "dtypes_tested": dtypes,
        "per_gpu": {},
        "fixed_shape_results": {},
    }

    for gpu_idx in range(resolved_gpu_count):
        gpu_key = f"gpu_{gpu_idx}"
        results["per_gpu"][gpu_key] = {}

        for dtype in dtypes:
            raw = None
            if dtype_outputs and dtype in dtype_outputs:
                raw = dtype_outputs[dtype]

            result = run_mamf_for_dtype(
                dtype, gpu_idx, raw_output=raw,
            )
            results["per_gpu"][gpu_key][dtype] = result

        # Fixed-shape check (using bf16)
        fixed_raw = None
        if fixed_shape_outputs and "bf16" in fixed_shape_outputs:
            fixed_raw = fixed_shape_outputs["bf16"]

        fixed_result = run_mamf_fixed_shape(
            "bf16", gpu_idx, raw_output=fixed_raw,
        )
        results["fixed_shape_results"][gpu_key] = fixed_result

    return results


class ComputeMatrixRunner(SectionRunner):
    """Section runner for compute benchmarks (Section 8.1).

    Parameters
    ----------
    gpu_count:
        Number of GPUs.
    include_tf32:
        Include TF32 tests.
    dtype_outputs:
        Pre-captured mamf-finder outputs for testing.
    fixed_shape_outputs:
        Pre-captured fixed-shape outputs for testing.
    """

    def __init__(
        self,
        gpu_count: int | None = None,
        *,
        include_tf32: bool = False,
        dtype_outputs: dict[str, str] | None = None,
        fixed_shape_outputs: dict[str, str] | None = None,
    ) -> None:
        super().__init__("compute")
        self._gpu_count = gpu_count
        self._include_tf32 = include_tf32
        self._dtype_outputs = dtype_outputs
        self._fixed_shape_outputs = fixed_shape_outputs

    def run(self) -> SectionResult:
        """Execute compute benchmark matrix."""
        started_at = datetime.now(timezone.utc).isoformat()

        try:
            metrics = collect_compute_matrix(
                gpu_count=self._gpu_count,
                include_tf32=self._include_tf32,
                dtype_outputs=self._dtype_outputs,
                fixed_shape_outputs=self._fixed_shape_outputs,
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
                error=f"Compute matrix failed: {exc}",
            )
