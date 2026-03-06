"""Tests for CLI help output and command discoverability.

Validates VAL-CLI-002: ornn-bench --help lists run, info, report, upload;
ornn-bench run --help documents scope/selective-run flags and output path options.
"""

from __future__ import annotations

import re

from typer.testing import CliRunner

from ornn_bench.cli import app

runner = CliRunner(env={"COLUMNS": "120"})


class TestTopLevelHelp:
    """Tests for the top-level --help output."""

    def test_help_exits_cleanly(self) -> None:
        """--help exits with code 0."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_lists_run_command(self) -> None:
        """Top-level help lists the 'run' command."""
        result = runner.invoke(app, ["--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "run" in clean_output.lower()

    def test_help_lists_info_command(self) -> None:
        """Top-level help lists the 'info' command."""
        result = runner.invoke(app, ["--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "info" in clean_output.lower()

    def test_help_lists_report_command(self) -> None:
        """Top-level help lists the 'report' command."""
        result = runner.invoke(app, ["--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "report" in clean_output.lower()

    def test_help_lists_upload_command(self) -> None:
        """Top-level help lists the 'upload' command."""
        result = runner.invoke(app, ["--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "upload" in clean_output.lower()

    def test_help_shows_description(self) -> None:
        """Top-level help includes the app description."""
        result = runner.invoke(app, ["--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "benchmark" in clean_output.lower()

    def test_version_flag(self) -> None:
        """--version outputs version and exits cleanly."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "0.2.0" in clean_output

    def test_no_args_shows_help(self) -> None:
        """Invoking with no arguments shows help (no_args_is_help)."""
        result = runner.invoke(app, [])
        # Typer's no_args_is_help shows help text but may exit with 0 or 2
        assert result.exit_code in (0, 2)
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "run" in clean_output.lower()


class TestRunCommandHelp:
    """Tests for the run command --help output."""

    def test_run_help_exits_cleanly(self) -> None:
        """run --help exits with code 0."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0

    def test_run_help_shows_compute_only_flag(self) -> None:
        """run --help documents --compute-only flag."""
        result = runner.invoke(app, ["run", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--compute-only" in clean_output

    def test_run_help_shows_memory_only_flag(self) -> None:
        """run --help documents --memory-only flag."""
        result = runner.invoke(app, ["run", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--memory-only" in clean_output

    def test_run_help_shows_interconnect_only_flag(self) -> None:
        """run --help documents --interconnect-only flag."""
        result = runner.invoke(app, ["run", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--interconnect-only" in clean_output

    def test_run_help_shows_output_option(self) -> None:
        """run --help documents --output / -o option for report path."""
        result = runner.invoke(app, ["run", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--output" in clean_output or "-o" in clean_output

    def test_run_help_shows_upload_flag(self) -> None:
        """run --help documents --upload flag."""
        result = runner.invoke(app, ["run", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--upload" in clean_output

    def test_run_help_shows_description(self) -> None:
        """run --help includes a description of what run does."""
        result = runner.invoke(app, ["run", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        output_lower = clean_output.lower()
        assert "benchmark" in output_lower or "run" in output_lower


class TestInfoCommandHelp:
    """Tests for the info command --help output."""

    def test_info_help_exits_cleanly(self) -> None:
        """info --help exits with code 0."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0

    def test_info_help_shows_description(self) -> None:
        """info --help includes a description."""
        result = runner.invoke(app, ["info", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        output_lower = clean_output.lower()
        assert "system" in output_lower or "environment" in output_lower or "gpu" in output_lower


class TestReportCommandHelp:
    """Tests for the report command --help output."""

    def test_report_help_exits_cleanly(self) -> None:
        """report --help exits with code 0."""
        result = runner.invoke(app, ["report", "--help"])
        assert result.exit_code == 0

    def test_report_help_shows_file_argument(self) -> None:
        """report --help documents the report file argument."""
        result = runner.invoke(app, ["report", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        output_lower = clean_output.lower()
        assert "file" in output_lower or "path" in output_lower or "json" in output_lower


class TestUploadCommandHelp:
    """Tests for the upload command --help output."""

    def test_upload_help_exits_cleanly(self) -> None:
        """upload --help exits with code 0."""
        result = runner.invoke(app, ["upload", "--help"])
        assert result.exit_code == 0

    def test_upload_help_shows_file_argument(self) -> None:
        """upload --help documents the report file argument."""
        result = runner.invoke(app, ["upload", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        output_lower = clean_output.lower()
        assert "file" in output_lower or "path" in output_lower or "json" in output_lower

    def test_upload_help_shows_api_key_option(self) -> None:
        """upload --help documents the --api-key option."""
        result = runner.invoke(app, ["upload", "--help"])
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--api-key" in clean_output


class TestAllCommandsCleanExit:
    """Ensure all help paths exit cleanly without crashes."""

    def test_all_commands_help_no_crash(self) -> None:
        """All command help paths return exit code 0."""
        commands = [
            ["--help"],
            ["run", "--help"],
            ["info", "--help"],
            ["report", "--help"],
            ["upload", "--help"],
        ]
        for cmd in commands:
            result = runner.invoke(app, cmd)
            assert result.exit_code == 0, (
                f"Command {cmd} failed with exit code {result.exit_code}: {result.output}"
            )
