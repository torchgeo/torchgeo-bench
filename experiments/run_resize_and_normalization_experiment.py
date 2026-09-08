#!/usr/bin/env python
"""Compare ResNet-18 input scaling and resize settings on m-eurosat.

Each job chooses ``input_normalization``, image size, and interpolation for raw inputs.

Usage:
    python experiments/run_resize_and_normalization_experiment.py
    python experiments/run_resize_and_normalization_experiment.py --devices 0 1 2
"""

import argparse
import sys

from _runner import Job, add_devices_argument, default_output, run_jobs

OUTPUT = default_output(__file__)

NORMALIZATIONS = ["bands_zscore", "none", "imagenet", "timm_default"]
IMAGE_SIZES: list[str] = ["null", "224", "256", "448", "512"]
INTERPOLATIONS = ["bilinear", "bicubic", "nearest"]


def build_jobs() -> list[Job]:
    """Build each setting combination, avoiding duplicate runs when resizing is disabled."""
    jobs: list[Job] = []
    for norm in NORMALIZATIONS:
        for size in IMAGE_SIZES:
            for interp in INTERPOLATIONS:
                if size == "null" and interp != "bilinear":
                    continue

                overrides = [
                    "model=timm/resnet18",
                    f"++model.input_normalization={norm}",
                    f"model.name=resnet18_{norm}",
                    "dataset.names=[m-eurosat]",
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
    """Run the input-scaling and resize comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_devices_argument(parser)
    args = parser.parse_args()
    return run_jobs(build_jobs(), args.devices, output=OUTPUT)


if __name__ == "__main__":
    sys.exit(main())
