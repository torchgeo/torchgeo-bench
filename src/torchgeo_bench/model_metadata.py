"""Centralized model metadata registry.

Maps result-CSV ``name`` values to display name, architecture family, and
exact encoder parameter count (measured via ``sum(p.numel() for p in
model.parameters())`` on each backbone with ``pretrained=False``/``weights=None``).

Architecture families:
  - ``"ViT"``         — pure Vision Transformer encoder
  - ``"CNN"``         — convolutional backbone
  - ``"Transformer"`` — hierarchical Transformer (e.g. Swin)
  - ``"—"``           — not a learned encoder (e.g. RCF random features)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMeta:
    display_name: str
    arch: str
    params_m: float | None


# Keys are the ``name`` column in result CSVs produced by the benchmark.
MODEL_METADATA: dict[str, ModelMeta] = {
    # --- geospatial foundation models (torchgeo) ---
    "tgeo_panopticon": ModelMeta(
        display_name="Panopticon",
        arch="ViT",
        params_m=98.12,   # panopticon_vitb14, verified
    ),
    "tgeo_dofa_base": ModelMeta(
        display_name="DOFA",
        arch="ViT",
        params_m=111.35,  # dofa_base_patch16_224, verified
    ),
    # --- OLMo Earth (multi-modal ViT encoder) ---
    "olmoearth_v1_1_base": ModelMeta(
        display_name="OLMo-Base",
        arch="ViT",
        params_m=276.38,  # OlmoEarthPretrain_v1 base v1.1, verified
    ),
    "olmoearth_v1_1_tiny": ModelMeta(
        display_name="OLMo-Tiny",
        arch="ViT",
        params_m=31.50,   # OlmoEarthPretrain_v1 tiny v1.1, verified
    ),
    # --- TerraTorch models ---
    "tt_clay_v1_5_base": ModelMeta(
        display_name="Clay V1.5",
        arch="ViT",
        params_m=92.10,   # timm_clay_v1_base, verified
    ),
    "tt_prithvi_eo_v2_300_tl": ModelMeta(
        display_name="Prithvi",
        arch="ViT",
        params_m=303.89,  # prithvi_eo_v2_300, verified
    ),
    "tt_terramind_v1_base": ModelMeta(
        display_name="TerraMind-MS",
        arch="ViT",
        params_m=85.54,   # terramind_v1_base, verified
    ),
    "tt_terramind_v1_base_rgb": ModelMeta(
        display_name="TerraMind-RGB",
        arch="ViT",
        params_m=85.54,   # same backbone as TerraMind-MS, RGB modality
    ),
    # --- timm ImageNet/DINOv3 models ---
    "vit_large_patch16_dinov3sat": ModelMeta(
        display_name="DINOv3-Sat",
        arch="ViT",
        params_m=303.08,  # vit_large_patch16_dinov3.sat493m, verified
    ),
    "vit_base_patch16_224": ModelMeta(
        display_name="ViT-B/16",
        arch="ViT",
        params_m=86.57,   # verified
    ),
    "vit_large_patch16_224": ModelMeta(
        display_name="ViT-L/16",
        arch="ViT",
        params_m=304.33,  # verified
    ),
    "resnet50": ModelMeta(
        display_name="ResNet-50",
        arch="CNN",
        params_m=25.56,   # verified
    ),
    "resnet18": ModelMeta(
        display_name="ResNet-18",
        arch="CNN",
        params_m=11.69,   # verified
    ),
    "convnext_large_dinov3": ModelMeta(
        display_name="ConvNeXt-L",
        arch="CNN",
        params_m=196.23,  # convnext_large.dinov3_lvd1689m, verified
    ),
    "mobilenetv3_large_100": ModelMeta(
        display_name="MobileNetV3",
        arch="CNN",
        params_m=5.48,    # verified
    ),
    "swin_tiny_patch4_window7_224": ModelMeta(
        display_name="Swin-Tiny",
        arch="Transformer",
        params_m=28.29,   # verified
    ),
    # --- baseline (no learned encoder) ---
    "rcf_empirical": ModelMeta(
        display_name="RCF",
        arch="—",
        params_m=None,
    ),
}
