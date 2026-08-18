"""Paper-sourced pretraining input sizes for the released encoders.

Each entry is the input resolution (pixels, square) the checkpoint was
pretrained at, traced to the original paper and released code rather than to
wrapper defaults — several wrappers disagree with their sources (Satlas
pretrains at 512 while the shipped transforms crop 256; timm's cfg for the
DINOv3 ConvNeXt reports a 224 default although pretraining ran the 256
regime). Full provenance with normalization constants and file-level
citations: ``experiments/scripts/slurm/resize_norm/pretrain_provenance.md``.

``None`` means no single pretraining size exists: OlmoEarth samples crops of
1--12 tokens per side at patch sizes 1--8 (up to 96 px, the value in
:data:`PRETRAIN_MAX_EXTENT`), and Scale-MAE conditions on ground-sample
distance and picks per-dataset sizes downstream.
"""

#: Checkpoint pretraining input size in pixels, keyed by result-row ``name``.
PRETRAIN_IMAGE_SIZES: dict[str, int | None] = {
    # DINOv3 (arXiv:2508.10104): 256 global crops; sat variant high-res 512.
    "convnext_large_dinov3": 256,
    "vit_large_patch16_dinov3": 256,
    "vit_large_patch16_dinov3sat": 256,
    # OlmoEarth (arXiv:2511.13655): trained across sizes; no single value.
    **dict.fromkeys(
        (
            "olmoearth_v1_nano",
            "olmoearth_v1_tiny",
            "olmoearth_v1_base",
            "olmoearth_v1_large",
            "olmoearth_v1_1_nano",
            "olmoearth_v1_1_tiny",
            "olmoearth_v1_1_base",
            "olmoearth_v1_2_nano",
            "olmoearth_v1_2_tiny",
            "olmoearth_v1_2_small",
            "olmoearth_v1_2_base",
        )
    ),
    # CROMA (arXiv:2311.00566): 60-180 px crops resized to 120.
    "tgeo_croma_base": 120,
    "tgeo_croma_large": 120,
    # DOFA (arXiv:2403.15356): 224, fixed (no pos-embed interpolation).
    "tgeo_dofa_base": 224,
    "tgeo_dofa_large": 224,
    # EarthLoc (arXiv:2403.06758): 320 (MixVPR fixed 20x20 grid).
    "tgeo_earthloc_s2_resnet50": 320,
    # Panopticon (arXiv:2503.10845): 224 global crops.
    "tgeo_panopticon": 224,
    # SatlasPretrain (arXiv:2211.15660): 512 for every backbone/modality.
    **dict.fromkeys(
        (
            "tgeo_resnet50_s2rgb_satlas_mi",
            "tgeo_resnet50_s2rgb_satlas_si",
            "tgeo_resnet152_s2rgb_satlas_mi",
            "tgeo_resnet152_s2rgb_satlas_si",
            "tgeo_swinv2b_naip_satlas_mi",
            "tgeo_swinv2b_naip_satlas_si",
            "tgeo_swinv2b_s2rgb_satlas_mi",
            "tgeo_swinv2b_s2rgb_satlas_si",
            "tgeo_swinv2t_s2rgb_satlas_mi",
            "tgeo_swinv2t_s2rgb_satlas_si",
        ),
        512,
    ),
    # SSL4EO-S12 MoCo (arXiv:2211.07044), SeCo (arXiv:2103.16607),
    # GASSL (arXiv:2011.09980): RandomResizedCrop(224) pipelines.
    "tgeo_resnet18_s2rgb_moco": 224,
    "tgeo_resnet50_s2rgb_moco": 224,
    "tgeo_resnet50_s2all_moco": 224,
    "tgeo_resnet18_s2rgb_seco": 224,
    "tgeo_resnet50_s2rgb_seco": 224,
    "tgeo_resnet50_fmow_gassl": 224,
    # Scale-MAE (arXiv:2212.14532): 224 encoder input, but GSD-conditioned
    # with per-dataset sizes downstream; no single comparable value.
    "tgeo_scalemae_large_fmow": None,
    "tgeo_scalemae_large_fmow_cls": None,
    # Clay v1 base (Clay-foundation/model tag v1.0): 224. The terratorch port
    # fixes a 256 grid, so the bench cannot realize this size (see provenance).
    "tt_clay_v1_5_base": 224,
    "tt_clay_v1_5_base_cls": 224,
    # Prithvi-EO (arXiv:2310.18660, arXiv:2412.02732): 224 (v2 tiles 256
    # randomly cropped to 224).
    **dict.fromkeys(
        (
            "tt_prithvi_eo_v1_100",
            "tt_prithvi_eo_v1_100_cls",
            "tt_prithvi_eo_v2_100_tl",
            "tt_prithvi_eo_v2_100_tl_cls",
            "tt_prithvi_eo_v2_300",
            "tt_prithvi_eo_v2_300_cls",
            "tt_prithvi_eo_v2_300_tl",
            "tt_prithvi_eo_v2_300_tl_cls",
            "tt_prithvi_eo_v2_600",
            "tt_prithvi_eo_v2_600_cls",
        ),
        224,
    ),
    # TerraMind v1 (arXiv:2504.11171): 224 crops of 264-px TerraMesh tiles.
    "tt_terramind_v1_base": 224,
    "tt_terramind_v1_base_rgb": 224,
    "tt_terramind_v1_large": 224,
    "tt_terramind_v1_large_rgb": 224,
}

#: Largest spatial extent seen in pretraining for size-agnostic models.
PRETRAIN_MAX_EXTENT: dict[str, int] = {
    name: 96 for name, size in PRETRAIN_IMAGE_SIZES.items() if name.startswith("olmoearth")
}
