#!/usr/bin/env python
"""Build reliability cache table from persisted UQ traces."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from torchgeo_bench.uq.reliability import build_reliability_frame

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True, help="Trace run directory (run_id=...).")
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


def _load_manifest(trace_dir: Path) -> pd.DataFrame:
    manifest_path = trace_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return manifest


def _load_trace(path: Path, fmt: str) -> pd.DataFrame:
    fmt_norm = fmt.strip().lower()
    if fmt_norm == "parquet":
        return pd.read_parquet(path)
    if fmt_norm == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported trace format in manifest: {fmt}")


def _run(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.trace_dir)
    output_path = args.out
    if output_path is None:
        output_path = args.trace_dir / "reliability_cache.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[pd.DataFrame] = []
    for entry in manifest.to_dict(orient="records"):
        trace_path = Path(str(entry["trace_path"]))
        fmt = str(entry.get("trace_format", "parquet"))
        if not trace_path.exists():
            logger.warning("Skipping missing trace file: %s", trace_path)
            continue

        trace_df = _load_trace(trace_path, fmt)
        if "confidence" not in trace_df.columns or "correct" not in trace_df.columns:
            logger.warning("Skipping trace missing confidence/correct columns: %s", trace_path)
            continue

        conf = pd.to_numeric(trace_df["confidence"], errors="coerce")
        corr = pd.to_numeric(trace_df["correct"], errors="coerce")
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
            "run_id",
            "model",
            "backbone",
            "name",
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
