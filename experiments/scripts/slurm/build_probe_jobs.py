"""Emit a job-list file for the SLURM probe sweep array.

Writes ``experiments/scripts/slurm/probe_sweep.jobs`` with one whitespace-separated
record per line:  ``<model_config> <dataset> <bands>``.

Usage::

    python experiments/scripts/slurm/build_probe_jobs.py            # all GeoFMs x classification + segmentation
    python experiments/scripts/slurm/build_probe_jobs.py --dry-run  # print and exit
    python experiments/scripts/slurm/build_probe_jobs.py --models terratorch/prithvi_eo_v2_300 \
        --datasets m-eurosat,m-forestnet --bands rgb,all

The output file is consumed by ``experiments/scripts/slurm/probe_sweep.sh``::

    sbatch --array=0-$(( $(wc -l < experiments/scripts/slurm/probe_sweep.jobs) - 1 )) \\
           experiments/scripts/slurm/probe_sweep.sh
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
    "torchgeo/resnet50_s2_all_moco": "all",
    # ``sam3_encoder`` only ships an RGB-pretrained patch embedding and has no
    # adapter path implemented yet — keep restricted.
    "sam3_encoder": "rgb",
    # DOFA empirically performs WORSE on multispectral than RGB across 11 of
    # 14 (variant, dataset) cells we tested (PR #85 comments), even though
    # the wrapper supports arbitrary wavelengths.  Keep RGB-only.
    "torchgeo/dofa_base": "rgb",
    "torchgeo/dofa_large": "rgb",
    # OlmoEarth has its own multi-modal sweep tooling
    # (``experiments/scripts/slurm/olmoearth_sweep.{py,sh}``) that constructs the right
    # 12-band / 10-band / RGB layout per dataset.  The build_probe_jobs.py
    # path only emits RGB-only entries to avoid passing
    # m-eurosat "all" (13 bands including swir_cirrus, which OlmoEarth's S2
    # modality doesn't accept) through this generic driver.
    "olmoearth_nano": "rgb",
    "olmoearth_tiny": "rgb",
    "olmoearth_base": "rgb",
    "olmoearth_large": "rgb",
    # All the remaining "RGB-pretrained" backbones can run on N-channel
    # input thanks to ``_adapt_first_conv`` (resnets / swins / scalemae) or
    # timm's native ``in_chans`` handling (dinov3 / dinov3sat).  The adapted
    # input conv is NOT the pretrained one — results on multispectral input
    # should be marked as "adapted*" rather than vanilla pretrained.
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
    # OlmoEarth (AI2 GeoFM) — all four variants.  Requires the
    # `[olmoearth]` extra (olmoearth-pretrain-minimal).  The original
    # disable note was about a transient HF download hang + the extra
    # not being installed in the cluster venv; both resolved.
    "olmoearth_nano",
    "olmoearth_tiny",
    "olmoearth_base",
    "olmoearth_large",
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
        default="experiments/scripts/slurm/probe_sweep.jobs",
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
