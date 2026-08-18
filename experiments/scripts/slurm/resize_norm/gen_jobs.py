#!/usr/bin/env python
"""Generate job lists for the crossed resize x normalization sweep.

Protocol grid (2 x 2), everything else fixed (bilinear, existing bands,
default partitions, kNN-5 + linear probe):

* normalization: bandspec_zscore | model_native   (dataset.normalization)
* resize: native | 224
    - native: dataset.image_size=null plus a per-wrapper override that turns
      off internal auto-resizing, so the image is never resized at all.
    - 224: dataset.image_size=224, wrapper behavior left as configured.

Clay v1.5 (fixed 32x32 pos_embed @ 256) and Scale-MAE (its own per-dataset
GSD/resize recipe via dataset_overrides) cannot honor a truly-native input;
they run all four cells anyway and are listed in fixed_pipeline.txt so the
analysis can exclude them from the resize contrast.

Outputs (TSV: name, cfg, norm, size, seed, datasets, extra-overrides):
    smoke.jobs    wrapper-class representatives x 3 datasets x 4 protocols
    primary.jobs  47 released encoders x 4 protocols (all available datasets)
    repeat.jobs   16 family representatives x 4 protocols x extra seeds
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent

NORMS = ["bandspec_zscore", "model_native"]
SIZES = ["null", "224"]

SMOKE_DATASETS = ["m-so2sat", "m-eurosat", "m-bigearthnet"]
# One model per wrapper/resize mechanism at risk: terratorch-fixed (clay),
# torchgeo auto_resize CNN-target-120 (croma), torchgeo auto_resize ViT (dofa),
# olmoearth min_image_size path, torchgeo swin window-256 (satlas), scale-mae
# GSD recipe, terratorch target-size ViT (prithvi), timm auto_resize ViT (dinov3).
SMOKE_MODELS = [
    "tt_clay_v1_5_base",
    "tgeo_croma_base",
    "tgeo_dofa_base",
    "olmoearth_v1_nano",
    "tgeo_swinv2t_s2rgb_satlas_mi",
    "tgeo_scalemae_large_fmow",
    "tt_prithvi_eo_v2_300",
    "vit_large_patch16_dinov3",
]

# Models whose pipeline resizes internally no matter what; no native override.
# Smoke-verified: CROMA (fixed 120 grid) and DOFA/Panopticon (fixed ViT
# pos_embed at 224) crash on truly-native inputs with auto_resize=false, so
# they keep their internal resize and are excluded from the resize contrast.
FIXED_PIPELINE = {
    "tt_clay_v1_5_base",
    "tgeo_scalemae_large_fmow",
    "tgeo_croma_base",
    "tgeo_croma_large",
    "tgeo_dofa_base",
    "tgeo_dofa_large",
    "tgeo_panopticon",
    # EarthLoc aggregates through a LayerNorm over a fixed 20x20=400-token
    # grid (320 px): native inputs crash at runtime.
    "tgeo_earthloc_s2_resnet50",
}
# Models where model_native normalization raises (no pretrain stats /
# expected_input_unit). Seeded by the smoke test (DOFA) and finalized from
# model_native_audit.tsv when present.
NO_MODEL_NATIVE = {"tgeo_dofa_base", "tgeo_dofa_large"}


def _load_audit() -> None:
    audit = HERE / "model_native_audit.tsv"
    if audit.exists():
        for row in audit.read_text().splitlines():
            name, status, err = row.split("\t", 2)
            # Only failures actually raised by the model_native strategy;
            # e.g. Scale-MAE's audit failure is a probe-size artifact.
            if status in ("init_fail", "forward_fail") and "model_native" in err:
                NO_MODEL_NATIVE.add(name)


# Scale-MAE's model-level dataset_overrides always win over dataset.image_size
# (main.py's effective_image_size), so its two resize cells would be byte-for-
# byte identical runs; it gets the normalization axis only, at its own recipe.
NORM_AXIS_ONLY = {"tgeo_scalemae_large_fmow"}
# OlmoEarth configs pin model.image_size=null, which also wins over
# dataset.image_size — the 224 cell must override the model-level knob.
OLMOEARTH_224 = ["model.image_size=224"]


# The OlmoEarth wrapper overrides normalize_inputs to identity and applies its
# own pretrained per-modality normalizer internally, so bandspec_zscore vs
# model_native are the same computation; primary/repeat run one labeled cell.
# (Smoke still runs both to confirm the collapse via feat_checksum.)
def norm_axis(name: str, cfg: str, smoke: bool) -> list[str]:
    if smoke:
        return NORMS
    # rcf: random features have no pretrain stats, model_native is undefined.
    if cfg.startswith("olmoearth") or name == "rcf" or name in NO_MODEL_NATIVE:
        return ["bandspec_zscore"]
    return NORMS


REPEAT_DATASETS = [
    "m-bigearthnet",
    "m-brick-kiln",
    "m-eurosat",
    "m-forestnet",
    "m-pv4ger",
    "m-so2sat",
    "treesatai",
]
REPEAT_REPS = {  # family -> representative
    "clay": "tt_clay_v1_5_base",
    "croma": "tgeo_croma_base",
    "dinov3": "vit_large_patch16_dinov3",
    "dofa": "tgeo_dofa_base",
    "moco": "tgeo_resnet50_s2rgb_moco",
    "olmoearth": "olmoearth_v1_base",
    "prithvi_eo_v1": "tt_prithvi_eo_v1_100",
    "prithvi_eo_v2": "tt_prithvi_eo_v2_300",
    "rcf": "rcf",
    "satlas": "tgeo_swinv2b_s2rgb_satlas_si",
    "scalemae": "tgeo_scalemae_large_fmow",
    "seco": "tgeo_resnet50_s2rgb_seco",
    "terramind": "tt_terramind_v1_base",
    "tgeo_earthloc_s2_resnet50": "tgeo_earthloc_s2_resnet50",
    "tgeo_panopticon": "tgeo_panopticon",
    "tgeo_resnet50_fmow_gassl": "tgeo_resnet50_fmow_gassl",
}


def native_overrides(name: str, cfg: str) -> list[str]:
    """Hydra overrides that make the model accept a never-resized input."""
    if name in FIXED_PIPELINE or name == "rcf" or cfg.startswith("olmoearth"):
        return []
    if cfg.startswith("torchgeo/"):
        return ["model.auto_resize=false"]
    if cfg.startswith("timm/"):
        return ["++model.auto_resize=false"]
    if cfg.startswith("terratorch/"):
        return ["model.target_size=null"]
    return []


def line(name, cfg, norm, size, seed, datasets, extra):
    return "\t".join([name, cfg, norm, size, str(seed), ",".join(datasets), " ".join(extra) or "-"])


def main() -> None:
    _load_audit()
    name2cfg = json.loads((HERE / "name2cfg.json").read_text())
    name2cfg["rcf"] = "rcf"
    models = json.loads((HERE / "models.json").read_text())

    bands: dict[str, list[str]] = {}
    for row in (HERE / "bands_map.tsv").read_text().splitlines():
        model, dataset, _ = row.split("\t")
        bands.setdefault(model, []).append(dataset)

    def protocols(name, smoke=False):
        cfg = name2cfg[name]
        sizes = ["null"] if name in NORM_AXIS_ONLY else SIZES
        for norm in norm_axis(name, cfg, smoke):
            for size in sizes:
                if size == "null":
                    extra = native_overrides(name, cfg)
                elif cfg.startswith("olmoearth"):
                    extra = OLMOEARTH_224
                else:
                    extra = []
                yield cfg, norm, size, extra

    smoke = [
        line(name, cfg, norm, size, 0, SMOKE_DATASETS, extra)
        for name in SMOKE_MODELS
        for cfg, norm, size, extra in protocols(name, smoke=True)
    ]
    primary = [
        line(name, cfg, norm, size, 0, bands[name], extra)
        for name in models
        for cfg, norm, size, extra in protocols(name)
    ]
    repeat = []
    for name in REPEAT_REPS.values():
        datasets = [d for d in REPEAT_DATASETS if d in bands.get(name, [])]
        seeds = range(5) if name == "rcf" else range(1, 5)
        for seed in seeds:
            for cfg, norm, size, extra in protocols(name):
                repeat.append(line(name, cfg, norm, size, seed, datasets, extra))

    (HERE / "smoke.jobs").write_text("\n".join(smoke) + "\n")
    (HERE / "primary.jobs").write_text("\n".join(primary) + "\n")
    (HERE / "repeat.jobs").write_text("\n".join(repeat) + "\n")
    (HERE / "fixed_pipeline.txt").write_text("\n".join(sorted(FIXED_PIPELINE)) + "\n")
    print(f"smoke={len(smoke)} primary={len(primary)} repeat={len(repeat)}")


if __name__ == "__main__":
    main()
