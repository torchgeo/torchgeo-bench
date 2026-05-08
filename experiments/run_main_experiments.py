#!/usr/bin/env python
"""Run every benchmark model across every dataset (or chosen subsets).

Ported from ``scripts/run_full_benchmark.py``. Each model is one job; the
job evaluates that model across all datasets specified by ``--datasets``
(default ``all``). Results land in ``results/all_results.csv``.

Usage:
    python experiments/run_main_experiments.py
    python experiments/run_main_experiments.py --devices 0 1 2 3
    python experiments/run_main_experiments.py --models timm/resnet18 timm/resnet50
    python experiments/run_main_experiments.py --datasets [m-eurosat,m-forestnet]
    python experiments/run_main_experiments.py --dry-run
"""

import argparse
import sys

from _runner import Job, add_devices_argument, run_jobs

OUTPUT = "results/all_results.csv"

MODELS = [
    "timm/convnext_base",
    "timm/convnext_large",
    "timm/convnext_large_dinov3",
    "timm/convnext_small",
    "timm/convnext_tiny",
    "timm/convnextv2_base",
    "timm/convnextv2_tiny",
    "timm/densenet121",
    "timm/densenet161",
    "timm/densenet169",
    "timm/densenet201",
    "timm/efficientnet_b0",
    "timm/efficientnet_b1",
    "timm/efficientnet_b2",
    "timm/efficientnet_b3",
    "timm/efficientnet_b4",
    "timm/efficientnetv2_l",
    "timm/efficientnetv2_m",
    "timm/efficientnetv2_s",
    "imagestats",
    "timm/maxvit_tiny_tf_224",
    "timm/mobilenetv3_large_100",
    "timm/mobilenetv3_small_100",
    "rcf",
    "timm/regnetx_002",
    "timm/regnetx_008",
    "timm/regnety_002",
    "timm/regnety_008",
    "timm/resnet101",
    "timm/resnet152",
    "timm/resnet18",
    "timm/resnet34",
    "timm/resnet50",
    "timm/vgg16",
    "timm/vgg19",
    "timm/wide_resnet50_2",
    "timm/vit/deit_base_patch16_224",
    "timm/vit/deit_small_patch16_224",
    "timm/vit/deit_tiny_patch16_224",
    "timm/vit/swin_base_patch4_window7_224",
    "timm/vit/swin_large_patch4_window7_224",
    "timm/vit/swin_small_patch4_window7_224",
    "timm/vit/swin_tiny_patch4_window7_224",
    "timm/vit/swinv2_base_window8_256",
    "timm/vit/swinv2_small_window8_256",
    "timm/vit/swinv2_tiny_window8_256",
    "timm/vit/vit_base_patch16_224",
    "timm/vit/vit_large_patch16_224",
    "timm/vit/vit_large_patch16_dinov3",
    "timm/vit/vit_large_patch16_dinov3sat",
    "timm/vit/vit_small_patch16_224",
    "timm/vit/vit_tiny_patch16_224",
    "torchgeo/resnet18_s2rgb_moco",
    "torchgeo/resnet18_s2rgb_seco",
    "torchgeo/resnet50_s2rgb_moco",
    "torchgeo/resnet50_s2rgb_seco",
    "torchgeo/resnet50_fmow_gassl",
    "torchgeo/resnet50_s2rgb_satlas_mi",
    "torchgeo/resnet50_s2rgb_satlas_si",
    "torchgeo/resnet152_s2rgb_satlas_mi",
    "torchgeo/resnet152_s2rgb_satlas_si",
    "torchgeo/swinv2b_naip_satlas_mi",
    "torchgeo/swinv2b_naip_satlas_si",
    "torchgeo/swinv2b_s2rgb_satlas_mi",
    "torchgeo/swinv2b_s2rgb_satlas_si",
    "torchgeo/swinv2t_s2rgb_satlas_mi",
    "torchgeo/swinv2t_s2rgb_satlas_si",
    "torchgeo/scalemae_large_fmow",
    "torchgeo/dofa_base",
    "torchgeo/dofa_large",
    "torchgeo/earthloc_s2_resnet50",
]


def build_jobs(models: list[str], datasets: str) -> list[Job]:
    """Build one job per model.

    Args:
        models: Model identifiers (Hydra ``model=`` overrides) to evaluate.
        datasets: Hydra value forwarded as ``dataset.names=<datasets>``
            (e.g. ``"all"`` or ``"[m-eurosat,m-forestnet]"``).
    """
    jobs: list[Job] = []
    for model in models:
        short = model.split("/")[-1]
        overrides = [f"model={model}", f"dataset.names={datasets}"]
        jobs.append(Job(label=short, overrides=overrides))
    return jobs


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Subset of models to run (default: all {len(MODELS)}).",
    )
    parser.add_argument(
        "--datasets",
        default="all",
        help="Hydra value for dataset.names — 'all' or a bracketed list "
        "such as '[m-eurosat,m-forestnet]'. Default: all.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT,
        help=f"Output CSV path (default: {OUTPUT}).",
    )
    add_devices_argument(parser)
    args = parser.parse_args()

    models = args.models if args.models is not None else MODELS
    jobs = build_jobs(models, args.datasets)
    return run_jobs(jobs, args.devices, output=args.output, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
