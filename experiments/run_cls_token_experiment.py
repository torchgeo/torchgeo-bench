#!/usr/bin/env python
"""CLS-token vs spatial-average sweep for ViT/DeiT models.

Ported from ``scripts/run_cls_token_experiment.sh``. Each ViT/DeiT model is
evaluated twice per dataset — once with ``model.use_cls_token=false``
(spatial average) and once with ``model.use_cls_token=true`` (CLS token).
Swin models are excluded (no CLS token). Results land in
``results/cls_token_experiment.csv``.

Usage:
    python experiments/run_cls_token_experiment.py
    python experiments/run_cls_token_experiment.py --devices 0 1 2
    python experiments/run_cls_token_experiment.py --dry-run
"""

import argparse
import sys

from _runner import Job, add_devices_argument, run_jobs

OUTPUT = "results/cls_token_experiment.csv"

DATASETS = ["m-eurosat", "m-so2sat"]

MODELS = [
    "timm/vit/vit_tiny_patch16_224",
    "timm/vit/vit_small_patch16_224",
    "timm/vit/vit_base_patch16_224",
    "timm/vit/vit_large_patch16_224",
    "timm/vit/vit_large_patch16_dinov3",
    "timm/vit/vit_large_patch16_dinov3sat",
    "timm/vit/deit_tiny_patch16_224",
    "timm/vit/deit_small_patch16_224",
    "timm/vit/deit_base_patch16_224",
]


def build_jobs() -> list[Job]:
    """Build dataset × model × use_cls_token jobs."""
    jobs: list[Job] = []
    for dataset in DATASETS:
        for model in MODELS:
            short = model.removeprefix("timm/vit/")
            for use_cls in (False, True):
                tag = "cls" if use_cls else "avg"
                overrides = [
                    f"model={model}",
                    f"model.use_cls_token={'true' if use_cls else 'false'}",
                    f"model.name={short}_{tag}",
                    f"dataset.names=[{dataset}]",
                    "dataset.partition=default",
                ]
                jobs.append(Job(label=f"{dataset} {short} {tag}", overrides=overrides))
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
