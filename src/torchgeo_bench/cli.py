"""Command-line interface for ``torchgeo-bench``.

Two subcommands:

- ``torchgeo-bench run [hydra overrides...]`` — runs the benchmark via Hydra.
- ``torchgeo-bench overfit-check [hydra overrides...]`` — pre-screens segmentation encoders.
- ``torchgeo-bench download {geobench_v1|geobench_v2|eurosat}`` — fetches data.

The ``run`` and ``overfit-check`` subcommands forward every remaining arg to Hydra
by mutating ``sys.argv`` and calling the respective Hydra-decorated entry point
in-process. We restore ``sys.argv`` afterwards so embedded use (tests, notebooks)
is safe.
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


def _overfit_check(hydra_args: list[str]) -> int:
    """Invoke the Hydra-decorated overfit-check entry point."""
    from torchgeo_bench.overfit_check import overfit_check as hydra_main

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


def main() -> int:
    """Entry point for the ``torchgeo-bench`` console script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Fast paths: pass raw args to Hydra, bypassing argparse.
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        return _run(sys.argv[2:])

    if len(sys.argv) > 1 and sys.argv[1] == "overfit-check":
        return _overfit_check(sys.argv[2:])

    # Standard argparse for download and --help.
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
        "overfit-check",
        help=(
            "Pre-screening sanity check: verify segmentation encoders can overfit "
            "a tiny training subset before running the full benchmark "
            "(forwards remaining args to Hydra)"
        ),
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

    args = parser.parse_args()
    if args.command == "download":
        return _download(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
