#!/usr/bin/env python3
"""Generate tests/fixtures/accuracy_baselines.csv from results/all_results.csv.

Usage::

    python scripts/update_baselines.py
    python scripts/update_baselines.py --output path/to/out.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_INPUT = _REPO_ROOT / "results" / "all_results.csv"
_DEFAULT_OUTPUT = _REPO_ROOT / "tests" / "fixtures" / "accuracy_baselines.csv"

TARGET_DATASETS = {"m-eurosat", "benv2", "so2sat", "m-pv4ger"}
TARGET_METHODS = {"knn5", "linear"}

# Pinned model names → Hydra config key (relative to conf/model/)
PINNED_CONFIGS: dict[str, str] = {
    "rcf": "rcf",
    "imagestats": "imagestats",
    "vit_large_patch16_dinov3sat": "timm/vit/vit_large_patch16_dinov3sat",
    "tgeo_croma_base": "torchgeo/croma_base",
    "tgeo_dofa_base": "torchgeo/dofa_base",
    "tgeo_panopticon": "torchgeo/panopticon",
    "tgeo_scalemae_large_fmow": "torchgeo/scalemae_large_fmow",
    "olmoearth_v1_nano": "olmoearth_nano",
    "tt_prithvi_eo_v2_300": "terratorch/prithvi_eo_v2_300",
    "tt_terramind_v1_base": "terratorch/terramind_v1_base",
    "tt_clay_v1_5_base": "terratorch/clay_v1_5",
    "tgeo_swinv2b_s2rgb_satlas_mi": "torchgeo/swinv2b_s2rgb_satlas_mi",
}

# Canonical bands string for each pinned model.
# For models with multiple band configs, select exactly one row per
# (name, dataset, method) triple for the fixture.
CANONICAL_BANDS: dict[str, str] = {
    "rcf": "all",
    "imagestats": "all",
    "vit_large_patch16_dinov3sat": "rgb",
    "tgeo_croma_base": "all",
    "tgeo_dofa_base": "all",
    "tgeo_panopticon": "all",
    "tgeo_scalemae_large_fmow": "all",
    "olmoearth_v1_nano": "rgb",
    "tt_prithvi_eo_v2_300": "all",
    "tt_terramind_v1_base": "all",
    "tt_clay_v1_5_base": "all",
    "tgeo_swinv2b_s2rgb_satlas_mi": "rgb",
}


def filter_and_deduplicate(
    df: pd.DataFrame,
    *,
    canonical_bands: dict[str, str],
    pinned_names: set[str],
    target_datasets: set[str] | None = None,
) -> pd.DataFrame:
    """Filter df to pinned models and deduplicate to one canonical bands row each.

    Args:
        df: Raw results DataFrame with columns including name, dataset, method,
            metric_name, bands, partition, metric_value, model.
        canonical_bands: Maps model name to the canonical bands string to keep.
        pinned_names: Set of model names to include.
        target_datasets: Restrict to these datasets; if None, all datasets pass.

    Returns:
        Tidy DataFrame with columns:
        model_config, name, dataset, method, metric_name, bands, partition, expected_value.
    """
    mask = (
        df["name"].isin(pinned_names)
        & (df["metric_name"] == "accuracy")
        & (df["method"].isin(TARGET_METHODS))
        & (df["partition"] == "default")
    )
    if target_datasets is not None:
        mask &= df["dataset"].isin(target_datasets)

    filtered = df[mask].copy()

    rows: list[dict] = []
    key_seen: set[tuple] = set()
    for name, group in filtered.groupby("name"):
        canon = canonical_bands.get(str(name))
        if canon is None:
            continue
        band_group = group[group["bands"] == canon]
        for _, row in band_group.iterrows():
            key = (str(name), row["dataset"], row["method"], row["metric_name"], row["bands"])
            if key in key_seen:
                continue
            key_seen.add(key)
            config = PINNED_CONFIGS.get(str(name), str(name))
            rows.append(
                {
                    "model_config": config,
                    "name": row["name"],
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "metric_name": row["metric_name"],
                    "bands": row["bands"],
                    "partition": row["partition"],
                    "expected_value": row["metric_value"],
                }
            )

    return pd.DataFrame(rows, columns=list(_output_columns()))


def _output_columns() -> list[str]:
    return [
        "model_config",
        "name",
        "dataset",
        "method",
        "metric_name",
        "bands",
        "partition",
        "expected_value",
    ]


def _diff_summary(old: pd.DataFrame | None, new: pd.DataFrame) -> None:
    key = ["name", "dataset", "method", "metric_name", "bands"]
    if old is None or old.empty:
        logger.info("Created %d rows (no previous fixture)", len(new))
        return
    old_keys = set(zip(*[old[c] for c in key]))
    new_keys = set(zip(*[new[c] for c in key]))
    added = new_keys - old_keys
    removed = old_keys - new_keys
    logger.info("Fixture diff: +%d added, -%d removed rows", len(added), len(removed))


def main(argv: list[str] | None = None) -> None:
    """Entry point for the update-baselines script."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=_DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    if not args.input.exists():
        logger.error("Input not found: %s", args.input)
        sys.exit(1)

    df = pd.read_csv(args.input)

    old: pd.DataFrame | None = None
    if args.output.exists():
        old = pd.read_csv(args.output)

    result = filter_and_deduplicate(
        df,
        canonical_bands=CANONICAL_BANDS,
        pinned_names=set(PINNED_CONFIGS.keys()),
        target_datasets=TARGET_DATASETS,
    )

    _diff_summary(old, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    logger.info("Wrote %d rows to %s", len(result), args.output)


if __name__ == "__main__":
    main()
