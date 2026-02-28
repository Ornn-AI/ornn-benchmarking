"""CLI entrypoint for ornn-bench.

Provides the Typer-based command hierarchy with ``run``, ``info``,
``report``, and ``upload`` sub-commands plus ``--version`` callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ornn_bench import __version__

# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print version and exit when ``--version`` is passed."""
    if value:
        typer.echo(f"ornn-bench {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Top-level app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="ornn-bench",
    help="Ornn GPU Benchmarking CLI — run standardized GPU benchmarks and compute scores.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Ornn GPU Benchmarking CLI — run standardized GPU benchmarks and compute scores."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    compute_only: Annotated[
        bool,
        typer.Option("--compute-only", help="Run only compute benchmarks."),
    ] = False,
    memory_only: Annotated[
        bool,
        typer.Option("--memory-only", help="Run only memory benchmarks."),
    ] = False,
    interconnect_only: Annotated[
        bool,
        typer.Option("--interconnect-only", help="Run only interconnect benchmarks."),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Path for the JSON report file."),
    ] = None,
    upload: Annotated[
        bool,
        typer.Option("--upload", help="Upload results to the Ornn API after the run."),
    ] = False,
) -> None:
    """Run the full GPU benchmark suite (or selected sections).

    Executes compute, memory, and interconnect benchmarks, computes Ornn-I
    and Ornn-T scores, displays a scorecard, and writes a JSON report.

    Use scope flags to run only a subset of benchmarks.
    """
    typer.echo("Benchmark run is not yet implemented.")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Display system and GPU environment information.

    Reports detected GPUs, driver/CUDA versions, and dependency status.
    Prints actionable remediation when required tools are missing.
    """
    typer.echo("Info command is not yet implemented.")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


@app.command()
def report(
    report_file: Annotated[
        Path,
        typer.Argument(help="Path to a JSON report file to display."),
    ],
) -> None:
    """Re-render a previously saved benchmark report.

    Reads the JSON report file and displays the scorecard in the terminal.
    Works on machines without GPU tools installed.
    """
    typer.echo("Report command is not yet implemented.")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


@app.command()
def upload(
    report_file: Annotated[
        Path,
        typer.Argument(help="Path to a JSON report file to upload."),
    ],
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            help="API key for authentication. Can also be set via ORNN_API_KEY env var.",
            envvar="ORNN_API_KEY",
        ),
    ] = None,
) -> None:
    """Upload a benchmark report to the Ornn API.

    Validates the report locally before uploading. Requires an API key
    provided via --api-key or the ORNN_API_KEY environment variable.
    """
    typer.echo("Upload command is not yet implemented.")
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def app_entry() -> None:
    """Package entrypoint wrapper for setuptools console_scripts."""
    app()
