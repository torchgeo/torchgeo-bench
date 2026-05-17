"""Emit a job-list file for the SLURM probe sweep array.

Writes ``scripts/slurm/probe_sweep.jobs`` with one whitespace-separated
record per line:  ``<model_config> <dataset> <bands>``.

Usage::

    python scripts/slurm/build_probe_jobs.py            # all GeoFMs x classification + segmentation
    python scripts/slurm/build_probe_jobs.py --dry-run  # print and exit
    python scripts/slurm/build_probe_jobs.py --models terratorch/prithvi_eo_v2_300 \
        --datasets m-eurosat,m-forestnet --bands rgb,all

The output file is consumed by ``scripts/slurm/probe_sweep.sh``::

    sbatch --array=0-$(( $(wc -l < scripts/slurm/probe_sweep.jobs) - 1 )) \\
           scripts/slurm/probe_sweep.sh
"""

import argparse
from pathlib import Path

# Default GeoFM lineup spanning all three new libraries plus the existing
# torchgeo wrappers. Each entry is the Hydra config name (without `.yaml`).
# Models whose pretrained weights are fixed at a single band-mode.  For
# these the (model, bands) cross-product needs to be filtered: feeding
# 3-channel RGB into a 13-channel ALL backbone (or vice versa) crashes
# because the input conv shape is baked in.  Keys are model config names,
# values are the only band modes that make sense for them.
SINGLE_BAND_MODE_MODELS: dict[str, str] = {
    "torchgeo/resnet50_s2rgb_moco": "rgb",
    "torchgeo/resnet50_s2_all_moco": "all",
    # RGB-only checkpoints (3-channel pretrained weights; all-mode crashes on
    # the weights' Normalize or the input conv).
    "torchgeo/resnet18_s2rgb_seco": "rgb",
    "torchgeo/resnet50_s2rgb_seco": "rgb",
    "torchgeo/resnet50_fmow_gassl": "rgb",
    "torchgeo/swinv2t_s2rgb_satlas_mi": "rgb",
    "torchgeo/swinv2t_s2rgb_satlas_si": "rgb",
    "torchgeo/swinv2b_s2rgb_satlas_mi": "rgb",
    "torchgeo/swinv2b_s2rgb_satlas_si": "rgb",
    "torchgeo/swinv2b_naip_satlas_mi": "rgb",
    "torchgeo/swinv2b_naip_satlas_si": "rgb",
    "sam3_encoder": "rgb",
    # OlmoEarth wrapper accepts 3-ch RGB or 12-ch S2 (no B10).  m-eurosat
    # all is 13-ch including cirrus; just sweep RGB for now.
    "olmoearth_base": "rgb",
    "olmoearth_large": "rgb",
    # 3-channel pretrained backbones (DOFA hardcodes S2-RGB wavelengths;
    # ScaleMAE-fMoW / EarthLoc are RGB-only).  Multi-band runs would crash
    # on the input conv shape mismatch.
    "torchgeo/dofa_base": "rgb",
    "torchgeo/dofa_large": "rgb",
    "torchgeo/scalemae_large_fmow": "rgb",
    "torchgeo/scalemae_large_fmow_cls": "rgb",
    "torchgeo/earthloc_s2_resnet50": "rgb",
    # DINOv3 ViT-Large pretrained on sat493m (RGB satellite imagery).
    "timm/vit/vit_large_patch16_dinov3sat": "rgb",
    # DINOv3 ViT-Large web-pretrained (RGB natural imagery baseline).
    "timm/vit/vit_large_patch16_dinov3": "rgb",
}


DEFAULT_MODELS: list[str] = [
    # terratorch
    "terratorch/prithvi_eo_v1_100",
    "terratorch/prithvi_eo_v2_100_tl",
    "terratorch/prithvi_eo_v2_300",
    "terratorch/prithvi_eo_v2_300_tl",
    "terratorch/prithvi_eo_v2_600",
    "terratorch/clay_v1_5",
    "terratorch/terramind_v1_base",
    "terratorch/terramind_v1_large",
    # torchgeo (main) GeoFM wrappers
    "torchgeo/croma_base",
    "torchgeo/croma_large",
    "torchgeo/panopticon",
    "torchgeo/dofa_base",
    "torchgeo/dofa_large",
    "torchgeo/scalemae_large_fmow",
    "torchgeo/resnet50_s2rgb_moco",
    "torchgeo/resnet50_s2_all_moco",
    "torchgeo/earthloc_s2_resnet50",
    # DINOv3 ViT-Large pretrained on satellite imagery (sat493m, RGB-only)
    "timm/vit/vit_large_patch16_dinov3sat",
    # DINOv3 ViT-Large web-pretrained (natural-image baseline, RGB-only)
    "timm/vit/vit_large_patch16_dinov3",
    # OlmoEarth (direct olmoearth-pretrain-minimal path) — disabled: HF
    # weight download hangs on the cluster; re-enable after caching weights
    # locally.
    # "olmoearth_base",
    # baselines
    "rcf",
    "imagestats",
]

# _cls variants of MAE-pretrained GeoFM backbones.  Pairs with the
# mean-pool defaults to give a CLS-vs-mean ablation per dataset.
# Terramind has no CLS token (architecture lacks one) so it's excluded.
CLS_VARIANT_MODELS: list[str] = [
    "terratorch/prithvi_eo_v1_100_cls",
    "terratorch/prithvi_eo_v2_100_tl_cls",
    "terratorch/prithvi_eo_v2_300_cls",
    "terratorch/prithvi_eo_v2_300_tl_cls",
    "terratorch/prithvi_eo_v2_600_cls",
    "terratorch/clay_v1_5_cls",
    "torchgeo/scalemae_large_fmow_cls",
]


DEFAULT_CLASSIFICATION_DATASETS: list[str] = [
    "m-eurosat",
    "m-forestnet",
    "m-so2sat",
    "m-pv4ger",
    "m-brick-kiln",
    "m-bigearthnet",
]

DEFAULT_V2_CLASSIFICATION_DATASETS: list[str] = [
    "benv2",
    "forestnet",
    "so2sat",
    "treesatai",
]

DEFAULT_SEGMENTATION_DATASETS: list[str] = [
    "burn_scars",
    "cloudsen12",
    "pastis",
    "spacenet2",
]


def _parse_csv(arg: str | None, default: list[str]) -> list[str]:
    if not arg:
        return default
    return [s.strip() for s in arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="comma-separated model config names")
    parser.add_argument("--datasets", help="comma-separated dataset names")
    parser.add_argument("--bands", default="rgb,all", help="comma-separated bands modes")
    parser.add_argument(
        "--normalizations",
        default="bandspec_zscore",
        help="comma-separated normalisation strategies",
    )
    parser.add_argument("--include-segmentation", action="store_true")
    parser.add_argument(
        "--include-cls-variants",
        action="store_true",
        help="Also emit _cls pooling variants for MAE backbones (CLS-vs-mean ablation).",
    )
    parser.add_argument(
        "--datasets-version",
        choices=["v1", "v2", "both"],
        default="v1",
        help="Which classification dataset family to sweep when --datasets isn't given.",
    )
    parser.add_argument(
        "--out",
        default="scripts/slurm/probe_sweep.jobs",
        help="output job-list file path",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = _parse_csv(args.models, DEFAULT_MODELS)
    if args.include_cls_variants:
        for m in CLS_VARIANT_MODELS:
            if m not in models:
                models.append(m)
    if args.datasets:
        datasets = _parse_csv(args.datasets, DEFAULT_CLASSIFICATION_DATASETS)
    else:
        datasets = []
        if args.datasets_version in ("v1", "both"):
            datasets += list(DEFAULT_CLASSIFICATION_DATASETS)
        if args.datasets_version in ("v2", "both"):
            datasets += list(DEFAULT_V2_CLASSIFICATION_DATASETS)
        if args.include_segmentation:
            datasets += DEFAULT_SEGMENTATION_DATASETS
    bands_modes = _parse_csv(args.bands, ["rgb", "all"])
    norms = _parse_csv(args.normalizations, ["bandspec_zscore"])

    # Datasets where every band IS the RGB subset (e.g. m-pv4ger NAIP RGB)
    # don't have a meaningful all-bands mode — drop the duplicate.
    rgb_only_datasets: set[str] = set()
    try:
        from torchgeo_bench.datasets import get_bench_dataset_class

        for d in datasets:
            try:
                cls = get_bench_dataset_class(d)
                if len(cls.bands) == len(cls.rgb_bands):
                    rgb_only_datasets.add(d)
            except (KeyError, AttributeError):
                pass
    except ImportError:
        pass

    lines: list[str] = []
    for m in models:
        forced = SINGLE_BAND_MODE_MODELS.get(m)
        per_model_modes = [forced] if forced else bands_modes
        for d in datasets:
            modes_d = ["rgb"] if d in rgb_only_datasets else per_model_modes
            for b in modes_d:
                for n in norms:
                    lines.append(f"{m} {d} {b} {n}")

    if args.dry_run:
        print("\n".join(lines))
        print(f"\n# {len(lines)} jobs")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs to {out}")


if __name__ == "__main__":
    main()
