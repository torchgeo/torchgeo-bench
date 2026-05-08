#!/usr/bin/env python
"""Sweep resnet18 on m-eurosat varying normalization, image size, and interpolation.

Results land in ``results/eurosat_effect_of_experimental_setting.csv``.

Usage:
    python experiments/run_resize_and_normalization_experiment.py
    python experiments/run_resize_and_normalization_experiment.py --devices 0 1 2
    python experiments/run_resize_and_normalization_experiment.py --dry-run
"""

import argparse
import sys

from _runner import Job, add_devices_argument, run_jobs

OUTPUT = "results/eurosat_effect_of_experimental_setting.csv"

NORMALIZATIONS = ["mean_stdev", "min_max", "percentile_2_98", "none"]
IMAGE_SIZES: list[str] = ["null", "224", "256", "448", "512"]
INTERPOLATIONS = ["bilinear", "bicubic", "nearest"]


def build_jobs() -> list[Job]:
    """Build the full norm × size × interpolation grid (skipping non-bilinear@null)."""
    jobs: list[Job] = []
    for norm in NORMALIZATIONS:
        for size in IMAGE_SIZES:
            for interp in INTERPOLATIONS:
                if size == "null" and interp != "bilinear":
                    continue

                overrides = [
                    "model=timm/resnet18",
                    "dataset.names=[m-eurosat]",
                    f"dataset.normalization={norm}",
                    f"dataset.image_size={size}",
                    "eval.merge_val=false",
                    "verbose=false",
                ]
                if size != "null":
                    overrides.append(f"dataset.interpolation={interp}")

                label = f"norm={norm} size={size} interp={interp}"
                jobs.append(Job(label=label, overrides=overrides))
    return jobs


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_devices_argument(parser)
    args = parser.parse_args()

    jobs = build_jobs()
    return run_jobs(jobs, args.devices, output=OUTPUT, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
