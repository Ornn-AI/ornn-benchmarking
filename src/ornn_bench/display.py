"""Scorecard and report display for terminal output.

Provides Rich-formatted scorecard rendering with Ornn-I, Ornn-T,
component metrics, qualification badges, and score status (VAL-CLI-007).

Also provides plain-text and JSON output renderers for non-TTY / piped
output modes (VAL-CLI-008, VAL-CLI-011).
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ornn_bench.models import (
    BenchmarkReport,
    Qualification,
    ScoreResult,
    ScoreStatus,
)


def _qualification_badge(qualification: Qualification | None) -> Text:
    """Return a colored qualification badge."""
    if qualification is None:
        return Text("N/A", style="dim")
    match qualification:
        case Qualification.PREMIUM:
            return Text("★ Premium", style="bold green")
        case Qualification.STANDARD:
            return Text("● Standard", style="bold yellow")
        case Qualification.BELOW:
            return Text("▼ Below", style="bold red")


def _score_text(value: float | None) -> Text:
    """Format a score value with color based on magnitude."""
    if value is None:
        return Text("—", style="dim")
    if value >= 90.0:
        style = "bold green"
    elif value >= 70.0:
        style = "bold yellow"
    else:
        style = "bold red"
    return Text(f"{value:.1f}", style=style)


def _status_text(status: ScoreStatus) -> Text:
    """Format score status with appropriate styling."""
    match status:
        case ScoreStatus.VALID:
            return Text("✓ Valid", style="green")
        case ScoreStatus.PARTIAL:
            return Text("⚠ Partial", style="yellow")
        case ScoreStatus.ERROR:
            return Text("✗ Error", style="red")


def render_scorecard(scores: ScoreResult, console: Console | None = None) -> None:
    """Render the scorecard to the console.

    Displays:
    - Ornn-I and Ornn-T scores with color coding
    - Component metric values
    - Qualification outcome (Premium/Standard/Below)
    - Score status (Valid/Partial/Error) with detail
    - Per-GPU breakdowns if multi-GPU
    - Aggregate method if applicable
    """
    if console is None:
        console = Console()

    # --- Main scores table ---
    score_table = Table(show_header=True, padding=(0, 2), expand=True)
    score_table.add_column("Metric", style="bold", min_width=12)
    score_table.add_column("Score", justify="right", min_width=10)

    score_table.add_row("Ornn-I (Inference)", _score_text(scores.ornn_i))
    score_table.add_row("Ornn-T (Training)", _score_text(scores.ornn_t))

    console.print(Panel(score_table, title="[bold]Ornn Scores[/bold]", border_style="blue"))

    # --- Component metrics table ---
    if scores.components:
        comp_table = Table(show_header=True, padding=(0, 2))
        comp_table.add_column("Component", style="bold")
        comp_table.add_column("Value", justify="right")

        component_labels = {
            "bw": "Memory Bandwidth (BW)",
            "fp8": "FP8 Compute (FP8)",
            "bf16": "BF16 Compute (BF16)",
            "ar": "All-Reduce (AR)",
        }
        for key, value in scores.components.items():
            label = component_labels.get(key, key)
            comp_table.add_row(label, f"{value:.4f}")

        console.print(
            Panel(comp_table, title="[bold]Component Metrics[/bold]", border_style="blue")
        )

    # --- Per-GPU breakdown (VAL-RUNBOOK-010) ---
    if scores.per_gpu_scores:
        gpu_table = Table(show_header=True, padding=(0, 2))
        gpu_table.add_column("GPU", style="bold")
        gpu_table.add_column("Ornn-I", justify="right")
        gpu_table.add_column("Ornn-T", justify="right")

        for gpu_record in scores.per_gpu_scores:
            gpu_table.add_row(
                gpu_record.gpu_uuid,
                _score_text(gpu_record.ornn_i),
                _score_text(gpu_record.ornn_t),
            )

        agg_label = f"Per-GPU Scores (aggregate: {scores.aggregate_method or 'N/A'})"
        console.print(
            Panel(gpu_table, title=f"[bold]{agg_label}[/bold]", border_style="blue")
        )

    # --- Qualification and status ---
    qual_table = Table(show_header=False, box=None, padding=(0, 2))
    qual_table.add_column("Key", style="bold", min_width=20)
    qual_table.add_column("Value")

    qual_table.add_row("Qualification", _qualification_badge(scores.qualification))
    qual_table.add_row("Score Status", _status_text(scores.score_status))

    if scores.score_status_detail:
        qual_table.add_row("Detail", Text(scores.score_status_detail, style="dim"))

    if scores.aggregate_method:
        qual_table.add_row("Aggregate Method", Text(scores.aggregate_method, style="dim"))

    console.print(
        Panel(qual_table, title="[bold]Qualification[/bold]", border_style="blue")
    )


def render_scorecard_plain(scores: ScoreResult) -> str:
    """Render scorecard as plain text for non-TTY / JSON output modes.

    Returns a clean string representation without ANSI codes.
    """
    lines: list[str] = []
    lines.append("=== Ornn Scores ===")
    lines.append(f"  Ornn-I (Inference): {_format_plain_score(scores.ornn_i)}")
    lines.append(f"  Ornn-T (Training):  {_format_plain_score(scores.ornn_t)}")
    lines.append("")

    if scores.components:
        lines.append("=== Component Metrics ===")
        component_labels = {
            "bw": "Memory Bandwidth (BW)",
            "fp8": "FP8 Compute (FP8)",
            "bf16": "BF16 Compute (BF16)",
            "ar": "All-Reduce (AR)",
        }
        for key, value in scores.components.items():
            label = component_labels.get(key, key)
            lines.append(f"  {label}: {value:.4f}")
        lines.append("")

    if scores.per_gpu_scores:
        lines.append(f"=== Per-GPU Scores (aggregate: {scores.aggregate_method or 'N/A'}) ===")
        for gpu_record in scores.per_gpu_scores:
            lines.append(
                f"  {gpu_record.gpu_uuid}: "
                f"Ornn-I={_format_plain_score(gpu_record.ornn_i)}, "
                f"Ornn-T={_format_plain_score(gpu_record.ornn_t)}"
            )
        lines.append("")

    lines.append("=== Qualification ===")
    qual_str = scores.qualification.value if scores.qualification else "N/A"
    lines.append(f"  Qualification: {qual_str}")
    lines.append(f"  Score Status: {scores.score_status.value}")
    if scores.score_status_detail:
        lines.append(f"  Detail: {scores.score_status_detail}")
    if scores.aggregate_method:
        lines.append(f"  Aggregate Method: {scores.aggregate_method}")

    return "\n".join(lines)


def _format_plain_score(value: float | None) -> str:
    """Format a score for plain text output."""
    if value is None:
        return "N/A"
    return f"{value:.1f}"


def render_report_plain(report: BenchmarkReport) -> str:
    """Render a full report as plain text for non-TTY / piped output.

    Returns a clean string representation without ANSI codes or
    box-drawing characters (VAL-CLI-008, VAL-CLI-011).
    """
    lines: list[str] = []

    # --- Report metadata ---
    lines.append("=== Report Metadata ===")
    lines.append(f"  Report ID: {report.report_id}")
    lines.append(f"  Created: {report.created_at}")
    lines.append(f"  Schema Version: {report.schema_version}")
    lines.append("")

    # --- Section summary ---
    if report.sections:
        lines.append("=== Sections ===")
        for section in report.sections:
            lines.append(f"  {section.name}: {section.status.value}")
        lines.append("")

    # --- Scorecard ---
    lines.append(render_scorecard_plain(report.scores))

    return "\n".join(lines)
