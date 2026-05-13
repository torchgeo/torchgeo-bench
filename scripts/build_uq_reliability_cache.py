#!/usr/bin/env python
"""Build reliability cache table from persisted UQ traces."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from torchgeo_bench.uq.reliability import build_reliability_frame
from torchgeo_bench.uq.traces import scan_traces

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True, help="Trace parquet dataset root.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output cache path (.parquet or .csv). Default: <trace-dir>/reliability_cache.parquet",
    )
    parser.add_argument("--bins", type=int, default=15, help="Number of reliability bins.")
    parser.add_argument(
        "--binning",
        type=str,
        default="equal_width",
        help="Binning strategy: equal_width or equal_mass.",
    )
    return parser


def _run(args: argparse.Namespace) -> int:
    trace_df = scan_traces(
        args.trace_dir,
        columns=[
            "trace_block_key",
            "run_id",
            "model",
            "backbone",
            "dataset",
            "partition",
            "bands",
            "normalization",
            "image_size",
            "interpolation",
            "uq_method",
            "corruption_type",
            "severity",
            "seed",
            "confidence",
            "correct",
        ],
    )
    output_path = args.out
    if output_path is None:
        output_path = args.trace_dir / "reliability_cache.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    if trace_df.empty:
        logger.warning("No trace rows were found under %s.", args.trace_dir)
        return 0

    group_cols = [
        "trace_block_key",
        "run_id",
        "model",
        "backbone",
        "dataset",
        "partition",
        "bands",
        "normalization",
        "image_size",
        "interpolation",
        "uq_method",
        "corruption_type",
        "severity",
        "seed",
    ]
    for key_vals, block_df in trace_df.groupby(group_cols, dropna=False):
        entry = dict(zip(group_cols, key_vals, strict=False))
        conf = pd.to_numeric(block_df["confidence"], errors="coerce")
        corr = pd.to_numeric(block_df["correct"], errors="coerce")
        valid = conf.notna() & corr.notna()
        if valid.sum() == 0:
            continue

        rel_df = build_reliability_frame(
            confidence=conf[valid].to_numpy(dtype=float),
            correct=corr[valid].to_numpy(dtype=float),
            bins=int(args.bins),
            binning=str(args.binning),
        )
        for col in [
            "trace_block_key",
            "run_id",
            "model",
            "backbone",
            "dataset",
            "partition",
            "bands",
            "normalization",
            "image_size",
            "interpolation",
            "uq_method",
            "corruption_type",
            "severity",
            "seed",
        ]:
            rel_df[col] = entry.get(col)
        rel_df["n_test"] = int(valid.sum())
        rel_df["binning"] = str(args.binning)
        rel_df["bins"] = int(args.bins)
        rows.append(rel_df)

    if not rows:
        logger.warning("No reliability rows were generated.")
        return 0

    output_df = pd.concat(rows, ignore_index=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        output_df.to_csv(output_path, index=False)
    else:
        output_df.to_parquet(output_path, index=False)
    logger.info("Wrote reliability cache: %s (%d rows)", output_path, len(output_df))
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
