"""Tests for packaging, entrypoint resolution, and distribution metadata.

Verifies that ``pip install ornn-bench`` exposes the production CLI
entrypoint correctly and that distribution metadata is release-ready.
"""

from __future__ import annotations

import re
import subprocess
import sys
from importlib.metadata import entry_points, metadata, version

# ---------------------------------------------------------------------------
# Entrypoint wiring
# ---------------------------------------------------------------------------


class TestConsoleScriptEntrypoint:
    """Verify the console_scripts entrypoint is registered and callable."""

    def test_entrypoint_registered(self) -> None:
        """ornn-bench console_scripts entry point should be discoverable."""
        eps = entry_points(group="console_scripts")
        matching = [ep for ep in eps if ep.name == "ornn-bench"]
        assert len(matching) == 1, "Expected exactly one ornn-bench entrypoint"
        ep = matching[0]
        assert ep.value == "ornn_bench.cli:app_entry"

    def test_entrypoint_importable(self) -> None:
        """The target callable must be importable without error."""
        from ornn_bench.cli import app_entry

        assert callable(app_entry)

    def test_version_via_cli(self) -> None:
        """``ornn-bench --version`` should print a semver-like string."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert re.search(r"\d+\.\d+\.\d+", result.stdout), (
            f"Expected semver in output: {result.stdout!r}"
        )

    def test_help_lists_subcommands(self) -> None:
        """``ornn-bench --help`` should list run, info, report, upload."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        for cmd in ("run", "info", "report", "upload"):
            assert cmd in result.stdout, f"Missing subcommand '{cmd}' in --help output"

    def test_python_m_invocation(self) -> None:
        """``python -m ornn_bench`` should work as an alternative entrypoint."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "ornn-bench" in result.stdout


# ---------------------------------------------------------------------------
# Distribution metadata
# ---------------------------------------------------------------------------


class TestDistributionMetadata:
    """Verify distribution metadata is ready for open-source release."""

    def test_package_name(self) -> None:
        """Package name should be 'ornn-bench'."""
        meta = metadata("ornn-bench")
        assert meta["Name"] == "ornn-bench"

    def test_version_matches_module(self) -> None:
        """Installed version should match ``ornn_bench.__version__``."""
        from ornn_bench import __version__

        installed = version("ornn-bench")
        assert installed == __version__

    def test_version_is_semver(self) -> None:
        """Version should follow semantic versioning."""
        ver = version("ornn-bench")
        assert re.match(r"^\d+\.\d+\.\d+", ver), f"Version {ver!r} is not semver"

    def test_description_non_empty(self) -> None:
        """Package should have a non-empty summary/description."""
        meta = metadata("ornn-bench")
        assert meta["Summary"], "Package summary is empty"

    def test_license_specified(self) -> None:
        """License field should be set."""
        meta = metadata("ornn-bench")
        assert meta["License"], "License is not specified"

    def test_requires_python(self) -> None:
        """Requires-Python should be set for >= 3.10."""
        meta = metadata("ornn-bench")
        req = meta["Requires-Python"]
        assert req is not None
        assert "3.10" in req

    def test_project_urls_present(self) -> None:
        """Project URLs (Homepage, Repository) should be populated."""
        meta = metadata("ornn-bench")
        urls = meta.get_all("Project-URL") or []
        url_text = " ".join(urls)
        assert "github.com" in url_text.lower(), (
            f"Expected GitHub URL in project-urls, got: {urls}"
        )

    def test_classifiers_include_benchmark(self) -> None:
        """Classifiers should include benchmark-related entries."""
        meta = metadata("ornn-bench")
        classifiers = meta.get_all("Classifier") or []
        assert any("Benchmark" in c for c in classifiers), (
            f"No Benchmark classifier found: {classifiers}"
        )

    def test_keywords_present(self) -> None:
        """Keywords should be populated for discoverability."""
        meta = metadata("ornn-bench")
        keywords = meta.get_all("Keywords") or meta.get("Keywords")
        # Keywords may be a comma-separated string or list
        assert keywords, "No keywords specified"

    def test_author_present(self) -> None:
        """Author should be specified."""
        meta = metadata("ornn-bench")
        author = meta.get("Author") or meta.get("Author-email")
        assert author, "Author not specified"


# ---------------------------------------------------------------------------
# Command routing
# ---------------------------------------------------------------------------


class TestCommandRouting:
    """Verify that the installed CLI routes to expected sub-commands."""

    def test_run_help(self) -> None:
        """``ornn-bench run --help`` should be routable and describe scope flags."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "run", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--compute-only" in result.stdout
        assert "--memory-only" in result.stdout
        assert "--interconnect-only" in result.stdout

    def test_info_help(self) -> None:
        """``ornn-bench info --help`` should be routable."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "info", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0

    def test_report_help(self) -> None:
        """``ornn-bench report --help`` should be routable."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "report", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0

    def test_upload_help(self) -> None:
        """``ornn-bench upload --help`` should be routable."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "upload", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0

    def test_unknown_command_fails(self) -> None:
        """Unknown sub-command should produce non-zero exit code."""
        result = subprocess.run(
            [sys.executable, "-m", "ornn_bench", "nonexistent"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode != 0
