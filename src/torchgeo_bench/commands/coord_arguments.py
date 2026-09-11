"""Argument registration for the coordinate benchmark command."""

import argparse


def add_coord_arguments(parser: argparse.ArgumentParser) -> None:
    """Register explicit coordinate flags without importing runtime dependencies."""
    parser.add_argument("--config", help="Coordinate YAML configuration")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Coordinate encoder preset")
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
    parser.add_argument(
        "--methods", choices=("knn", "linear"), nargs="+", default=argparse.SUPPRESS
    )
    parser.add_argument("--split", choices=("random", "spatial", "both"), default=argparse.SUPPRESS)
    parser.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cell-deg", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--knn-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--knn-device", default=argparse.SUPPRESS)
    parser.add_argument("--device", default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--output", default=argparse.SUPPRESS, help="Result CSV path")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print validated YAML without running"
    )
