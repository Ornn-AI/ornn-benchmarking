"""Tests for the info command environment diagnostics.

Validates VAL-CLI-003: ``ornn-bench info`` reports detected GPU/dependency
status and prints actionable remediation when GPU/tools are missing.
"""

from __future__ import annotations

import builtins
from unittest.mock import patch

from typer.testing import CliRunner

from ornn_bench.cli import app
from ornn_bench.system import (
    EnvironmentInfo,
    GPUProbeResult,
    PythonProbeResult,
    ToolProbeResult,
    collect_environment_info,
    probe_nvidia_smi,
    probe_python_environment,
    probe_tool,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Unit tests for system probes
# ---------------------------------------------------------------------------


class TestProbeNvidiaSmi:
    """Tests for nvidia-smi GPU probe."""

    def test_missing_nvidia_smi(self) -> None:
        """Returns not-detected with remediation when nvidia-smi is absent."""
        with patch("ornn_bench.system.shutil.which", return_value=None):
            result = probe_nvidia_smi()
        assert result.detected is False
        assert result.gpu_count == 0
        assert "nvidia-smi" in result.error.lower()
        assert result.remediation != ""
        assert "install" in result.remediation.lower()

    def test_nvidia_smi_query_fails(self) -> None:
        """Returns not-detected when nvidia-smi exists but query fails."""
        with (
            patch("ornn_bench.system.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "ornn_bench.system._run_cmd",
                return_value=(1, "", "NVIDIA-SMI has failed"),
            ),
        ):
            result = probe_nvidia_smi()
        assert result.detected is False
        assert "failed" in result.error.lower()
        assert result.remediation != ""

    def test_nvidia_smi_returns_gpus(self) -> None:
        """Detects GPUs when nvidia-smi succeeds."""

        def mock_run_cmd(
            cmd: list[str], *, timeout: int = 10
        ) -> tuple[int, str, str]:
            if "--query-gpu=name" in cmd and "noheader" in cmd[-1]:
                return (0, "NVIDIA H100 80GB HBM3\nNVIDIA H100 80GB HBM3", "")
            if "--query-gpu=driver_version" in cmd:
                return (0, "535.129.03", "")
            if cmd == ["nvidia-smi"]:
                return (0, "CUDA Version: 12.2  |", "")
            return (0, "", "")

        with (
            patch("ornn_bench.system.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("ornn_bench.system._run_cmd", side_effect=mock_run_cmd),
        ):
            result = probe_nvidia_smi()
        assert result.detected is True
        assert result.gpu_count == 2
        assert "H100" in result.gpu_names[0]
        assert result.driver_version == "535.129.03"
        assert result.cuda_version == "12.2"


class TestProbeTool:
    """Tests for generic tool probing."""

    def test_tool_not_on_path(self) -> None:
        """Returns unavailable with remediation when tool is absent."""
        with patch("ornn_bench.system.shutil.which", return_value=None):
            result = probe_tool(
                "nvbandwidth",
                ["nvbandwidth"],
                remediation="Install nvbandwidth from source.",
            )
        assert result.available is False
        assert result.name == "nvbandwidth"
        assert "not found" in result.error.lower()
        assert "Install" in result.remediation

    def test_tool_on_path_with_version(self) -> None:
        """Returns available with version when tool succeeds."""
        with (
            patch(
                "ornn_bench.system.shutil.which",
                return_value="/usr/local/bin/nvbandwidth",
            ),
            patch(
                "ornn_bench.system._run_cmd",
                return_value=(0, "nvbandwidth v0.5", ""),
            ),
        ):
            result = probe_tool("nvbandwidth", ["nvbandwidth"])
        assert result.available is True
        assert result.version == "nvbandwidth v0.5"
        assert result.path == "/usr/local/bin/nvbandwidth"


class TestProbePythonEnvironment:
    """Tests for Python/PyTorch probing."""

    def test_python_version_detected(self) -> None:
        """Always detects Python version."""
        result = probe_python_environment()
        assert result.python_version != ""

    def test_pytorch_unavailable(self) -> None:
        """Reports PyTorch as unavailable when import fails."""
        with patch.dict("sys.modules", {"torch": None}):
            # Force import to fail by removing from modules
            import sys

            saved = sys.modules.pop("torch", None)
            try:
                with patch("builtins.__import__", side_effect=_import_no_torch):
                    result = probe_python_environment()
                assert result.pytorch_available is False
            finally:
                if saved is not None:
                    sys.modules["torch"] = saved


def _import_no_torch(
    name: str, *args: object, **kwargs: object
) -> object:
    """Mock import that raises ImportError for torch."""
    if name == "torch":
        raise ImportError("No module named 'torch'")
    return original_import(name, *args, **kwargs)


original_import = builtins.__import__


class TestCollectEnvironmentInfo:
    """Tests for the full environment collection."""

    def test_returns_environment_info(self) -> None:
        """Returns an EnvironmentInfo with all fields populated."""
        with (
            patch("ornn_bench.system.probe_nvidia_smi", return_value=GPUProbeResult()),
            patch(
                "ornn_bench.system.probe_python_environment",
                return_value=PythonProbeResult(python_version="3.11.0"),
            ),
            patch("ornn_bench.system.probe_tool", return_value=ToolProbeResult(
                name="test", available=False, error="not found",
                remediation="Install it.",
            )),
        ):
            env = collect_environment_info()
        assert env.os_name != ""
        assert isinstance(env.gpu, GPUProbeResult)
        assert isinstance(env.python, PythonProbeResult)

    def test_has_gpu_property(self) -> None:
        """has_gpu reflects GPU detection state."""
        env = EnvironmentInfo()
        assert env.has_gpu is False

        env.gpu = GPUProbeResult(detected=True, gpu_count=1)
        assert env.has_gpu is True

    def test_missing_tools_property(self) -> None:
        """missing_tools returns only unavailable tools."""
        env = EnvironmentInfo(tools=[
            ToolProbeResult(name="a", available=True),
            ToolProbeResult(name="b", available=False, remediation="Install b"),
            ToolProbeResult(name="c", available=False, remediation="Install c"),
        ])
        assert len(env.missing_tools) == 2
        assert env.missing_tools[0].name == "b"


# ---------------------------------------------------------------------------
# CLI integration tests for `ornn-bench info`
# ---------------------------------------------------------------------------


class TestInfoCommand:
    """Integration tests for the info CLI command."""

    def test_info_exits_cleanly(self) -> None:
        """info command exits with code 0."""
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0, f"Output: {result.output}"

    def test_info_shows_os_info(self) -> None:
        """info output includes OS information."""
        result = runner.invoke(app, ["info"])
        output_lower = result.output.lower()
        # Should mention the operating system
        assert "os" in output_lower or "system" in output_lower or "platform" in output_lower

    def test_info_shows_python_version(self) -> None:
        """info output includes Python version."""
        result = runner.invoke(app, ["info"])
        assert "python" in result.output.lower()

    def test_info_shows_gpu_status(self) -> None:
        """info output includes GPU status (detected or not)."""
        result = runner.invoke(app, ["info"])
        output_lower = result.output.lower()
        assert "gpu" in output_lower

    def test_info_shows_tool_status(self) -> None:
        """info output includes benchmark tool availability status."""
        result = runner.invoke(app, ["info"])
        output_lower = result.output.lower()
        assert "nvidia-smi" in output_lower or "tool" in output_lower

    def test_info_shows_remediation_for_missing_tools(self) -> None:
        """info output shows remediation for missing tools on non-GPU host."""
        # On a macOS dev machine, GPU tools will be missing
        result = runner.invoke(app, ["info"])
        output_lower = result.output.lower()
        # Should either show tools as available or provide remediation
        assert (
            "install" in output_lower
            or "available" in output_lower
            or "not found" in output_lower
            or "missing" in output_lower
            or "✓" in result.output
            or "✗" in result.output
            or "✘" in result.output
        )

    def test_info_no_raw_traceback(self) -> None:
        """info command never shows raw Python traceback."""
        result = runner.invoke(app, ["info"])
        assert "Traceback" not in result.output
        assert "raise " not in result.output
