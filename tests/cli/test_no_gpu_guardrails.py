"""Tests for no-GPU run guardrails.

Validates VAL-CLI-010: On systems without an NVIDIA GPU, ``ornn-bench run``
exits gracefully with clear guidance and no crash.
"""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from ornn_bench.cli import app
from ornn_bench.system import check_gpu_available

runner = CliRunner()


class TestCheckGpuAvailable:
    """Unit tests for the GPU availability check."""

    def test_no_nvidia_smi_on_path(self) -> None:
        """Returns (False, message) when nvidia-smi is not on PATH."""
        with patch("ornn_bench.system.shutil.which", return_value=None):
            available, message = check_gpu_available()
        assert available is False
        assert "nvidia-smi" in message.lower()
        assert "install" in message.lower() or "driver" in message.lower()

    def test_nvidia_smi_fails(self) -> None:
        """Returns (False, message) when nvidia-smi query fails."""
        with (
            patch("ornn_bench.system.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "ornn_bench.system._run_cmd",
                return_value=(1, "", "NVIDIA-SMI has failed"),
            ),
        ):
            available, message = check_gpu_available()
        assert available is False
        assert "failed" in message.lower()

    def test_nvidia_smi_no_gpus(self) -> None:
        """Returns (False, message) when nvidia-smi returns empty GPU list."""
        with (
            patch("ornn_bench.system.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch("ornn_bench.system._run_cmd", return_value=(0, "", "")),
        ):
            available, _message = check_gpu_available()
        assert available is False

    def test_nvidia_smi_has_gpus(self) -> None:
        """Returns (True, message) when GPUs are detected."""
        with (
            patch("ornn_bench.system.shutil.which", return_value="/usr/bin/nvidia-smi"),
            patch(
                "ornn_bench.system._run_cmd",
                return_value=(0, "NVIDIA H100 80GB HBM3", ""),
            ),
        ):
            available, message = check_gpu_available()
        assert available is True
        assert "H100" in message

    def test_message_includes_guidance(self) -> None:
        """Failure message includes actionable guidance."""
        with patch("ornn_bench.system.shutil.which", return_value=None):
            _, message = check_gpu_available()
        # Should mention what user can do
        assert "info" in message.lower() or "report" in message.lower()
        # Should mention driver installation
        assert "driver" in message.lower() or "install" in message.lower()

    def test_message_mentions_alternative_commands(self) -> None:
        """Failure message suggests alternative commands that work without GPU."""
        with patch("ornn_bench.system.shutil.which", return_value=None):
            _, message = check_gpu_available()
        assert "ornn-bench info" in message or "info" in message.lower()


class TestRunCommandNoGpuGuardrail:
    """Integration tests for the run command's no-GPU guardrail."""

    def test_run_exits_gracefully_without_gpu(self) -> None:
        """run command exits with non-zero code when no GPU is present."""
        with patch(
            "ornn_bench.cli.check_gpu_available",
            return_value=(False, "No NVIDIA GPU detected."),
        ):
            result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_run_shows_guidance_without_gpu(self) -> None:
        """run command shows human-readable guidance when no GPU is present."""
        guidance = (
            "No NVIDIA GPU detected: nvidia-smi not found.\n"
            "Install NVIDIA drivers to run benchmarks."
        )
        with patch(
            "ornn_bench.cli.check_gpu_available",
            return_value=(False, guidance),
        ):
            result = runner.invoke(app, ["run"])
        output_lower = result.output.lower()
        assert "gpu" in output_lower or "nvidia" in output_lower

    def test_run_no_raw_traceback_without_gpu(self) -> None:
        """run command does not show raw Python traceback when GPU is absent."""
        with patch(
            "ornn_bench.cli.check_gpu_available",
            return_value=(False, "No GPU found."),
        ):
            result = runner.invoke(app, ["run"])
        assert "Traceback" not in result.output
        assert "raise " not in result.output

    def test_run_with_scope_flags_still_checks_gpu(self) -> None:
        """Scope flags (--compute-only etc.) still trigger GPU check."""
        with patch(
            "ornn_bench.cli.check_gpu_available",
            return_value=(False, "No GPU found."),
        ):
            for flag in ["--compute-only", "--memory-only", "--interconnect-only"]:
                result = runner.invoke(app, ["run", flag])
                assert result.exit_code != 0, f"Flag {flag} should still fail without GPU"

    def test_run_proceeds_when_gpu_available(self) -> None:
        """run command proceeds past GPU check when GPU is present.

        Since benchmarks aren't implemented yet, it will still exit
        with code 1 but the output should NOT contain GPU-missing messages.
        """
        with patch(
            "ornn_bench.cli.check_gpu_available",
            return_value=(True, "Found 1 GPU(s)"),
        ):
            result = runner.invoke(app, ["run"])
        # Should not show "no gpu" type messages
        assert "nvidia-smi not found" not in result.output.lower()

    def test_run_exit_code_is_nonzero_without_gpu(self) -> None:
        """Exit code is non-zero (specifically 1) when GPU is absent."""
        with patch(
            "ornn_bench.cli.check_gpu_available",
            return_value=(False, "No GPU."),
        ):
            result = runner.invoke(app, ["run"])
        assert result.exit_code == 1
