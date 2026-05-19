"""Command-line interface for ``torchgeo-bench``.

Two subcommands:

- ``torchgeo-bench run [hydra overrides...]`` — runs the benchmark via Hydra.
- ``torchgeo-bench uq [hydra overrides...]`` — runs UQ benchmark via Hydra.
- ``torchgeo-bench download {geobench_v1|geobench_v2|eurosat}`` — fetches data.

The ``run`` and ``uq`` subcommands forward every remaining arg to Hydra by
mutating ``sys.argv`` and calling the corresponding entry point in-process.
We restore ``sys.argv`` afterwards so embedded use (tests, notebooks) is safe.
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


<<<<<<< HEAD
def _uq(hydra_args: list[str]) -> int:
    """Invoke the Hydra-decorated UQ main, restoring ``sys.argv`` afterwards.

    Args:
        hydra_args: Hydra override arguments forwarded to the UQ pipeline.

    Returns:
        Process exit code.
    """
    from torchgeo_bench.uq.pipeline import main as hydra_main

    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *hydra_args]
        hydra_main()
    finally:
        sys.argv = saved
    return 0


def _download(args: argparse.Namespace) -> int:
=======
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
>>>>>>> main
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


<<<<<<< HEAD
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchgeo-bench",
        description="Lightweight benchmarking framework for geospatial foundation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "run",
        help="Run benchmark experiments (forwards remaining args to Hydra)",
        add_help=False,
    )
    sub.add_parser(
        "uq",
        help="Run UQ benchmark experiments (forwards remaining args to Hydra)",
        add_help=False,
    )

    dl = sub.add_parser(
        "download",
        help="Download benchmark datasets into ./data/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dl.add_argument(
        "target",
        choices=["geobench_v1", "geobench_v2", "eurosat"],
        help="What to download.",
    )
    dl.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Benchmark data root.",
    )
    dl.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="(geobench_v2 only) Comma-separated dataset names. Defaults to all "
        "benchmark-supported V2 datasets.",
    )
    return parser


def main() -> int:
    """Entry point for the ``torchgeo-bench`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return _run(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "uq":
        return _uq(sys.argv[2:])

    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "download":
        return _download(args)
    parser.print_help()
    return 1
=======
def main() -> None:
    """Entry point for the ``torchgeo-bench`` console script."""
    app()
>>>>>>> main


if __name__ == "__main__":
    main()
