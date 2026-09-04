"""Download command handler."""

import argparse
from pathlib import Path

from torchgeo_bench.commands._common import setup_logging


def download(args: argparse.Namespace) -> None:
    """Download one of the supported dataset collections."""
    from torchgeo_bench.commands._download_runtime import (
        download_eurosat,
        download_geobench_v1,
        download_geobench_v2,
        download_resisc45,
    )

    setup_logging(verbose=True)
    names = None
    if args.datasets is not None:
        names = [name.strip() for name in args.datasets.split(",") if name.strip()]
        if not names:
            raise SystemExit("error: --datasets must contain at least one dataset name")
    output_dir = Path(args.output_dir)
    if args.target == "geobench_v1":
        download_geobench_v1(output_dir, datasets=names)
    elif args.target == "geobench_v2":
        download_geobench_v2(output_dir, datasets=names)
    else:
        if names is not None:
            raise SystemExit("error: --datasets is only supported for GeoBench downloads")
        if args.target == "eurosat":
            download_eurosat(output_dir)
        else:
            download_resisc45(output_dir)
