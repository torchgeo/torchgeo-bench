#!/usr/bin/env python3
"""Download and extract the GeoBench dataset from Hugging Face.

This is a standalone script wrapper around src.download module.
For CLI usage, prefer: torchgeo-bench download
"""

import argparse
import logging
from pathlib import Path

from src.download import GEOBENCH_V2_DATASETS, download_geobench_v1, download_geobench_v2

logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and extract the GeoBench dataset from Hugging Face.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="GeoBench version to download",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="For v2: comma-separated dataset names or 'all' (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/",
        help="Directory to download and extract the dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.force:
        print("Force mode enabled: existing files will be re-downloaded")

    if args.version == "v1":
        download_geobench_v1(args.output_dir, args.force)
    elif args.version == "v2":
        # Parse datasets argument
        datasets = None
        if args.datasets and args.datasets != "all":
            datasets = [d.strip() for d in args.datasets.split(",")]
        download_geobench_v2(args.output_dir, datasets, args.force)
