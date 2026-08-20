"""Argument registration for the supervised coordinate prior command."""

import argparse

from torchgeo_bench.coordbench.prior_config import PRIOR_NAMES


def add_coord_prior_arguments(parser: argparse.ArgumentParser) -> None:
    """Register prior flags without importing estimators or datasets."""
    parser.add_argument("--config", help="Coordinate prior YAML configuration")
    parser.add_argument(
        "--dataset",
        "--datasets",
        "--name",
        "--names",
        dest="datasets",
        nargs="+",
        default=argparse.SUPPRESS,
        help="Benchmark names or families, or all",
    )
    parser.add_argument("--methods", choices=PRIOR_NAMES, nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--split", choices=("random", "spatial", "both"), default=argparse.SUPPRESS)
    parser.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cell-deg", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--grid-cell-size", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--smoothing", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--nearest-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--nearest-weights", choices=("uniform", "distance"), default=argparse.SUPPRESS
    )
    parser.add_argument("--kde-bandwidth", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--output", default=argparse.SUPPRESS, help="Prior result CSV path")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print validated YAML without running"
    )
