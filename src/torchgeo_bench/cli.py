"""Command-line interface for ``torchgeo-bench``.

Two subcommands:

- ``torchgeo-bench run [hydra overrides...]`` — runs the benchmark via Hydra.
- ``torchgeo-bench uq [hydra overrides...]`` — runs UQ benchmark via Hydra.
- ``torchgeo-bench nf [hydra overrides...]`` — runs NF stage-1 pipeline via Hydra.
- ``torchgeo-bench sample-size [hydra overrides...]`` — runs sample-size calibration sweep.
- ``torchgeo-bench download {geobench_v1|geobench_v2|eurosat}`` — fetches data.

The ``run``, ``uq``, ``nf``, and ``sample-size`` subcommands forward every remaining
arg to Hydra by mutating ``sys.argv`` and calling the corresponding entry point
in-process.  We restore ``sys.argv`` afterwards so embedded use (tests, notebooks)
is safe.
"""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)],
)

app = typer.Typer(
    name="torchgeo-bench",
    help="Lightweight benchmarking framework for geospatial foundation models.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run benchmark experiments (extra args forwarded to Hydra).",
)
def run(ctx: typer.Context) -> None:
    """Run benchmark experiments; extra args are forwarded to Hydra."""
    from torchgeo_bench.main import main as hydra_main

    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *ctx.args]
        hydra_main()
    finally:
        sys.argv = saved


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run UQ benchmark experiments (extra args forwarded to Hydra).",
)
def uq(ctx: typer.Context) -> None:
    """Run UQ benchmark experiments; extra args are forwarded to Hydra."""
    from torchgeo_bench.uq.pipeline import main as hydra_main

    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *ctx.args]
        hydra_main()
    finally:
        sys.argv = saved


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run NF stage-1 pipeline (extra args forwarded to Hydra).",
)
def nf(ctx: typer.Context) -> None:
    """Run NF stage-1 Optuna pipeline; extra args are forwarded to Hydra."""
    from torchgeo_bench.nf_pipeline import main as hydra_main

    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *ctx.args]
        hydra_main()
    finally:
        sys.argv = saved


@app.command(
    name="sample-size",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    help="Run sample-size calibration sweep (extra args forwarded to Hydra).",
)
def sample_size(ctx: typer.Context) -> None:
    """Run sample-size calibration sweep; extra args are forwarded to Hydra."""
    from torchgeo_bench.sample_size_pipeline import main as hydra_main

    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *ctx.args]
        hydra_main()
    finally:
        sys.argv = saved


@app.command(help="Download benchmark datasets.")
def download(
    target: Annotated[
        str,
        typer.Argument(help="What to download: geobench_v1 | geobench_v2 | eurosat"),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Benchmark data root."),
    ] = Path("data"),
    datasets: Annotated[
        str | None,
        typer.Option(help="(geobench_v2 only) Comma-separated dataset names."),
    ] = None,
) -> None:
    """Download a benchmark dataset to disk."""
    from torchgeo_bench.download import (
        download_eurosat,
        download_geobench_v1,
        download_geobench_v2,
    )

    valid = {"geobench_v1", "geobench_v2", "eurosat"}
    if target not in valid:
        typer.echo(f"Unknown target {target!r}. Choose from: {', '.join(sorted(valid))}", err=True)
        raise typer.Exit(1)

    if target == "geobench_v1":
        download_geobench_v1(output_dir)
    elif target == "geobench_v2":
        names = [n.strip() for n in datasets.split(",") if n.strip()] if datasets else None
        download_geobench_v2(output_dir, datasets=names)
    elif target == "eurosat":
        download_eurosat(output_dir)


def main() -> None:
    """Entry point for the ``torchgeo-bench`` console script."""
    app()


if __name__ == "__main__":
    main()
