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
from ornn_bench.display import render_report_plain, render_scorecard
from ornn_bench.models import BenchmarkReport
from ornn_bench.runner import RunOrchestrator, build_section_runners
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

    # Determine scope from flags
    scope: set[str] | None = None
    scope_flags = {
        "compute": compute_only,
        "memory": memory_only,
        "interconnect": interconnect_only,
    }
    active_scopes = {name for name, flag in scope_flags.items() if flag}
    if active_scopes:
        scope = active_scopes

    # Build runners and orchestrator
    runners = build_section_runners()

    def _on_progress(section: str, status: str) -> None:
        if status == "started":
            console.print(f"  [bold]▶[/bold] Running [cyan]{section}[/cyan]...")
        elif status == "completed":
            console.print(f"  [green]✓[/green] {section} [green]completed[/green]")
        elif status == "skipped":
            console.print(f"  [dim]-[/dim] {section} [dim]skipped[/dim]")
        elif status == "failed":
            console.print(f"  [red]✗[/red] {section} [red]failed[/red]")
        elif status == "timeout":
            console.print(f"  [yellow]⏱[/yellow] {section} [yellow]timed out[/yellow]")

    console.print(Panel("[bold]Starting benchmark run[/bold]", border_style="blue"))

    orch = RunOrchestrator(runners=runners, scope=scope, on_progress=_on_progress)
    report = orch.execute()

    # Write JSON report
    if output is None:
        output = Path(f"ornn_report_{report.report_id[:8]}.json")
    output.write_text(report.model_dump_json(indent=2))
    console.print(f"\n  Report saved to [bold]{output}[/bold]")

    # Summary
    total = len(report.sections)
    completed = sum(1 for s in report.sections if s.status.value == "completed")
    failed = sum(1 for s in report.sections if s.status.value in ("failed", "timeout"))
    skipped = sum(1 for s in report.sections if s.status.value == "skipped")

    summary_parts = [f"[green]{completed} completed[/green]"]
    if failed:
        summary_parts.append(f"[red]{failed} failed[/red]")
    if skipped:
        summary_parts.append(f"[dim]{skipped} skipped[/dim]")
    console.print(f"\n  {' · '.join(summary_parts)} of {total} sections\n")

    # Display scorecard (VAL-CLI-007)
    render_scorecard(report.scores, console=console)

    if orch.has_failures:
        console.print(
            Panel(
                "[yellow]Some sections failed. Review the report for details.[/yellow]",
                title="[bold yellow]Partial Failure[/bold yellow]",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=2)

    console.print(
        Panel(
            "[green]All sections completed successfully.[/green]",
            title="[bold green]Run Complete[/bold green]",
            border_style="green",
        )
    )


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
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output the report as machine-readable JSON (no ANSI, no progress).",
        ),
    ] = False,
    plain: Annotated[
        bool,
        typer.Option(
            "--plain",
            help="Output the report as plain text (no ANSI codes or box-drawing).",
        ),
    ] = False,
) -> None:
    """Re-render a previously saved benchmark report.

    Reads the JSON report file and displays the scorecard in the terminal.
    Works on machines without GPU tools installed.

    Use --json for machine-readable JSON output (piped/non-TTY safe).
    Use --plain for clean text output without ANSI codes.
    """
    if not report_file.exists():
        if json_output:
            typer.echo('{"error": "Report file not found"}')
        elif plain:
            typer.echo(f"Error: Report file not found: {report_file}")
        else:
            console.print(f"[red]Error:[/red] Report file not found: {report_file}")
        raise typer.Exit(code=1)

    try:
        raw = report_file.read_text()
        bench_report = BenchmarkReport.model_validate_json(raw)
    except Exception as exc:
        if json_output:
            typer.echo(f'{{"error": "Failed to parse report: {exc}"}}')
        elif plain:
            typer.echo(f"Error: Failed to parse report: {exc}")
        else:
            console.print(f"[red]Error:[/red] Failed to parse report: {exc}")
        raise typer.Exit(code=1) from None

    # --- JSON output mode (VAL-CLI-011) ---
    if json_output:
        typer.echo(bench_report.model_dump_json(indent=2))
        return

    # --- Plain text output mode (VAL-CLI-011) ---
    if plain:
        typer.echo(render_report_plain(bench_report))
        return

    # --- Rich terminal output (default) ---

    # Display report metadata
    meta_table = Table(show_header=False, box=None, padding=(0, 2))
    meta_table.add_column("Key", style="bold")
    meta_table.add_column("Value")
    meta_table.add_row("Report ID", bench_report.report_id)
    meta_table.add_row("Created", bench_report.created_at)
    meta_table.add_row("Schema Version", bench_report.schema_version)
    console.print(
        Panel(meta_table, title="[bold]Report Metadata[/bold]", border_style="blue")
    )

    # Section summary
    if bench_report.sections:
        section_table = Table(show_header=True, padding=(0, 2))
        section_table.add_column("Section", style="bold")
        section_table.add_column("Status")
        for section in bench_report.sections:
            status_style = {
                "completed": "green",
                "failed": "red",
                "timeout": "yellow",
                "skipped": "dim",
                "pending": "dim",
                "running": "cyan",
            }.get(section.status.value, "")
            section_table.add_row(
                section.name,
                f"[{status_style}]{section.status.value}[/{status_style}]",
            )
        console.print(
            Panel(section_table, title="[bold]Sections[/bold]", border_style="blue")
        )

    # Scorecard display (VAL-CLI-007, VAL-CLI-008)
    render_scorecard(bench_report.scores, console=console)


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
