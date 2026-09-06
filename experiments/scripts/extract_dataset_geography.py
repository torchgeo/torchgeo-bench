"""Generate the committed per-dataset geographic store.

Thin CLI over :mod:`torchgeo_bench.geography`, which owns all the extraction
logic.  Writes one ``<dataset>.json`` per registered dataset plus an
``index.json`` under ``docs/_static/_dataset_geography/``.

Coverage comes from the dataset registry, so a newly registered dataset is
picked up here automatically -- no edit to this script is needed.

Usage::

    # regenerate the whole store
    python experiments/scripts/extract_dataset_geography.py --all

    # regenerate a single dataset without rescanning everything
    python experiments/scripts/extract_dataset_geography.py --dataset m-eurosat

    # report coverage without touching the store
    python experiments/scripts/extract_dataset_geography.py --check
"""

from __future__ import annotations

import argparse
import logging
import os
import warnings

from torchgeo_bench.datasets import list_datasets
from torchgeo_bench.geography import (
    STORE_DIR,
    build_index,
    extract_geography,
    list_geography,
    missing_datasets,
    write_record,
)


def _check() -> int:
    """Print the current store's coverage; non-zero if a dataset is missing."""
    store = list_geography()
    missing = missing_datasets()

    for name in sorted(store):
        record = store[name]
        detail = record.reason or f"n={record.n}"
        print(f"  {name:20s} {record.status:15s} {detail}")

    if missing:
        print(f"\n{len(missing)} registered dataset(s) with no record: {sorted(missing)}")
        print("Run with --all (or --dataset <name>) to generate them.")
        return 1
    print(f"\nAll {len(store)} registered datasets have a record.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="regenerate every registered dataset")
    group.add_argument("--dataset", help="regenerate a single dataset by name")
    group.add_argument("--check", action="store_true", help="report coverage, write nothing")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, (os.cpu_count() or 8)),
        help="processes used for the V1 metadata scan",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    warnings.filterwarnings("ignore")

    if args.check:
        return _check()

    names = list_datasets() if args.all else [args.dataset]
    if not args.all and args.dataset not in list_datasets():
        print(f"Unknown dataset {args.dataset!r}. Available: {', '.join(list_datasets())}")
        return 1

    for name in names:
        record = extract_geography(name, workers=args.workers)
        write_record(record)
        detail = record.reason or f"n={record.n}, {len(record.bins)} bins"
        print(f"  {name:20s} {record.status:15s} {detail}")

    index = build_index()
    totals = index["totals"]
    print(
        f"\nWrote {STORE_DIR}: {totals['datasets']} records "
        f"({totals['extracted']} extracted, {totals['samples']} samples)"
    )
    for continent, share in list(totals["continents"].items())[:6]:
        print(f"  {continent:20s} {share:5.1f}%")

    missing = missing_datasets()
    if missing:
        print(f"\nWARNING: no record for {sorted(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
