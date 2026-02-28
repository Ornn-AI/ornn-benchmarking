"""CLI entrypoint for ornn-bench."""

import typer

app = typer.Typer(
    name="ornn-bench",
    help="Ornn GPU Benchmarking CLI — run standardized GPU benchmarks and compute scores.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the ornn-bench version."""
    from ornn_bench import __version__

    typer.echo(f"ornn-bench {__version__}")


def app_entry() -> None:
    """Package entrypoint wrapper for setuptools console_scripts."""
    app()
