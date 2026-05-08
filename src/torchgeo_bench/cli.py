"""Command-line interface for ``torchgeo-bench``.

Two subcommands:

- ``torchgeo-bench run [hydra overrides...]`` — runs the benchmark via Hydra.
- ``torchgeo-bench download {geobench_v1|geobench_v2|eurosat}`` — fetches data.

The ``run`` subcommand forwards every remaining arg to Hydra by mutating
``sys.argv`` and calling :func:`torchgeo_bench.main.main` in-process. We
restore ``sys.argv`` afterwards so embedded use (tests, notebooks) is safe.
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(hydra_args: list[str]) -> int:
    """Invoke the Hydra-decorated benchmark main, restoring argv afterwards."""
    from torchgeo_bench.main import main as hydra_main

    saved = sys.argv[:]
    try:
        sys.argv = [saved[0], *hydra_args]
        hydra_main()
    finally:
        sys.argv = saved
    return 0


def _download(args: argparse.Namespace) -> int:
    from torchgeo_bench.download import (
        download_eurosat,
        download_geobench_v1,
        download_geobench_v2,
    )

    output = Path(args.output_dir)
    if args.target == "geobench_v1":
        download_geobench_v1(output)
    elif args.target == "geobench_v2":
        names = (
            [n.strip() for n in args.datasets.split(",") if n.strip()] if args.datasets else None
        )
        download_geobench_v2(output, datasets=names)
    elif args.target == "eurosat":
        download_eurosat(output)
    else:  # pragma: no cover — argparse choices guard this
        raise AssertionError(f"Unknown download target: {args.target}")
    return 0


def run_command(args: argparse.Namespace) -> int:
    """Execute the run command."""
    import subprocess

    # Pass through all arguments after 'run'
    hydra_args = args.hydra_args if args.hydra_args else []

    # Run the script directly with subprocess to preserve Hydra's argument handling
    try:
        cmd = [sys.executable, "-m", "torchgeo_bench"] + hydra_args
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        return 1


def overfit_check_command(args: argparse.Namespace) -> int:
    """Execute the overfit-check command."""
    import subprocess

    hydra_args = args.hydra_args if args.hydra_args else []
    try:
        cmd = [sys.executable, "-m", "torchgeo_bench.overfit_check"] + hydra_args
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        logger.error(f"Overfit check failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    # Special handling for "run" command - pass everything after "run" to Hydra
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Extract hydra args (everything after "run")
        hydra_args = sys.argv[2:]

        # Create a minimal args object for run_command
        args = argparse.Namespace(hydra_args=hydra_args)
        return run_command(args)

    # Special handling for "overfit-check" command
    if len(sys.argv) > 1 and sys.argv[1] == "overfit-check":
        hydra_args = sys.argv[2:]
        args = argparse.Namespace(hydra_args=hydra_args)
        return overfit_check_command(args)

    # For other commands, use standard argparse
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


    # # Overfit-check command
    # subparsers.add_parser(
    #     "overfit-check",
    #     help=(
    #         "Pre-screening sanity check: verify segmentation encoders can overfit "
    #         "a tiny training subset before running the full benchmark"
    #     ),
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    # )

def main() -> int:
    """Entry point for the ``torchgeo-bench`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return _run(sys.argv[2:])

    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "download":
        return download_command(args)
    elif args.command in ("run", "overfit-check"):
        raise AssertionError("This should never be reached due to special handling above.")
    else:
        parser.print_help()
        return 1
        return _download(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
