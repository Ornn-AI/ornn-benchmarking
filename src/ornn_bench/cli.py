"""CLI entrypoint for ornn-bench.

Provides the Typer-based command hierarchy with ``run``, ``info``,
``report``, and ``upload`` sub-commands plus ``--version`` callback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ornn_bench import __version__
from ornn_bench.system import check_gpu_available, collect_environment_info

console = Console()

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
    gpu_available, gpu_message = check_gpu_available()
    if not gpu_available:
        console.print(
            Panel(
                gpu_message,
                title="[bold red]No NVIDIA GPU Detected[/bold red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    # GPU is present — benchmark orchestration not yet implemented.
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
    env = collect_environment_info()

    # --- System section ---
    system_table = Table(show_header=False, box=None, padding=(0, 2))
    system_table.add_column("Key", style="bold")
    system_table.add_column("Value")
    system_table.add_row("OS", env.os_name)
    system_table.add_row("Kernel", env.kernel_version)
    system_table.add_row("CPU", env.cpu_model)
    console.print(Panel(system_table, title="[bold]System[/bold]", border_style="blue"))

    # --- GPU section ---
    if env.has_gpu:
        gpu = env.gpu
        gpu_table = Table(show_header=False, box=None, padding=(0, 2))
        gpu_table.add_column("Key", style="bold")
        gpu_table.add_column("Value")
        gpu_table.add_row("GPU Count", str(gpu.gpu_count))
        for i, name in enumerate(gpu.gpu_names):
            gpu_table.add_row(f"GPU {i}", name)
        gpu_table.add_row("Driver", gpu.driver_version or "unknown")
        gpu_table.add_row("CUDA", gpu.cuda_version or "unknown")
        console.print(Panel(gpu_table, title="[bold green]GPU[/bold green]", border_style="green"))
    else:
        gpu_msg = env.gpu.error or "No NVIDIA GPU detected"
        remediation = env.gpu.remediation
        body = f"[yellow]{gpu_msg}[/yellow]"
        if remediation:
            body += f"\n\n[dim]{remediation}[/dim]"
        console.print(Panel(body, title="[bold yellow]GPU[/bold yellow]", border_style="yellow"))

    # --- Python section ---
    py = env.python
    py_table = Table(show_header=False, box=None, padding=(0, 2))
    py_table.add_column("Key", style="bold")
    py_table.add_column("Value")
    py_table.add_row("Python", py.python_version)
    if py.pytorch_available:
        py_table.add_row("PyTorch", py.pytorch_version)
        cuda_status = (
            f"available (CUDA {py.pytorch_cuda_version})"
            if py.pytorch_cuda_available
            else "CPU only"
        )
        py_table.add_row("PyTorch CUDA", cuda_status)
    else:
        py_table.add_row("PyTorch", "[yellow]not installed[/yellow]")
    console.print(Panel(py_table, title="[bold]Python[/bold]", border_style="blue"))

    # --- Tools section ---
    tools_table = Table(show_header=True, padding=(0, 2))
    tools_table.add_column("Tool", style="bold")
    tools_table.add_column("Status")
    tools_table.add_column("Details")
    for tool in env.tools:
        if tool.available:
            status = "[green]✓ available[/green]"
            details = tool.version or tool.path
        else:
            status = "[red]✗ missing[/red]"
            details = tool.remediation or tool.error
        tools_table.add_row(tool.name, status, details)
    console.print(
        Panel(tools_table, title="[bold]Benchmark Tools[/bold]", border_style="blue")
    )


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
