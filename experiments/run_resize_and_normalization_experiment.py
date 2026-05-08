#!/usr/bin/env python
"""Sweep resnet18 on m-eurosat varying image size, interpolation, and the model's input normalization mode.

Sweeps the model-side ``input_normalization`` knob (``bands_zscore``,
``none``, ``imagenet``, ``timm_default``) — the dataset always emits raw
values, so normalization is configured on the model.

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
    """Build the full norm × size × interpolation grid (skipping non-bilinear@null)."""
    jobs: list[Job] = []
    for norm in NORMALIZATIONS:
        for size in IMAGE_SIZES:
            for interp in INTERPOLATIONS:
                if size == "null" and interp != "bilinear":
                    continue

                overrides = [
                    "model=timm/resnet18",
                    f"model.input_normalization={norm}",
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
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_devices_argument(parser)
    args = parser.parse_args()
    return run_jobs(build_jobs(), args.devices, output=OUTPUT)


if __name__ == "__main__":
    sys.exit(main())
