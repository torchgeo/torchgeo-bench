"""torchgeo foundation-model wrappers for torchgeo-bench.

Each wrapper class loads a torchgeo pretrained model and exposes the
``BenchModel`` interface (``forward_patch_features`` returning ``(B, K)``).
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import BenchModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_torchgeo_factory(factory_name: str):
    """Return the model-factory function from ``torchgeo.models``."""
    import torchgeo.models as tgm

    fn = getattr(tgm, factory_name, None)
    if fn is None:
        raise ValueError(f"torchgeo.models has no factory function '{factory_name}'")
    return fn


def _resolve_torchgeo_weights(weights_class_name: str, weights_member: str):
    """Return the concrete weights enum member.

    E.g. ``_resolve_torchgeo_weights("ResNet18_Weights", "SENTINEL2_RGB_MOCO")``
    returns ``torchgeo.models.ResNet18_Weights.SENTINEL2_RGB_MOCO``.
    """
    import torchgeo.models as tgm

    cls = getattr(tgm, weights_class_name, None)
    if cls is None:
        raise ValueError(f"torchgeo.models has no weights class '{weights_class_name}'")
    member = getattr(cls, weights_member, None)
    if member is None:
        raise ValueError(f"{weights_class_name} has no member '{weights_member}'")
    return member


def _auto_resize(images: torch.Tensor, target_size: int) -> torch.Tensor:
    h, w = images.shape[-2], images.shape[-1]
    if h != target_size or w != target_size:
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="bicubic",
            align_corners=False,
        )
    return images


def _extract_normalize_transforms(weights) -> nn.Sequential | None:
    """Extract only the ``Normalize`` layers from a torchgeo weights transform.

    The full ``weights.transforms`` typically includes ``Resize``,
    ``CenterCrop``, and ``Normalize`` steps.  We handle spatial transforms
    separately (via ``auto_resize``/``target_size``), so this helper pulls
    out just the normalization layers to apply in the forward pass.
    """
    if not hasattr(weights, "transforms") or weights.transforms is None:
        return None
    transform = weights.transforms
    if callable(transform) and not isinstance(transform, nn.Module):
        transform = transform()

    from torchvision.transforms import Normalize as NormalizeV1

    try:
        from torchvision.transforms.v2 import Normalize as NormalizeV2
    except ImportError:
        NormalizeV2 = NormalizeV1  # type: ignore[misc,assignment]

    norms = [t for t in transform if isinstance(t, (NormalizeV1, NormalizeV2))]
    if not norms:
        return None
    return nn.Sequential(*norms)


# ---------------------------------------------------------------------------
# ResNet (timm backbone loaded via torchgeo)
# ---------------------------------------------------------------------------


class TorchGeoResNetBench(BenchModel):
    """Wrapper for torchgeo ResNet models (resnet18 / resnet50 / resnet152).

    These return ``timm.models.resnet.ResNet`` instances.  We replace ``.fc``
    with ``Identity()`` to get headless (B, K) feature vectors.
    """

    def __init__(
        self,
        num_channels: int,
        factory: str = "resnet50",
        weights_class: str = "ResNet50_Weights",
        weights_member: str = "SENTINEL2_RGB_MOCO",
        auto_resize: bool = False,
        target_size: int | None = 224,
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_channels=num_channels)
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.backbone = _resolve_torchgeo_factory(factory)(weights=weights)
        self.backbone.fc = nn.Identity()
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.input_norm = _extract_normalize_transforms(weights)

    @torch.no_grad()
    def forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return headless ResNet embeddings of shape ``(B, K)``."""
        del bboxes
        if self.input_norm is not None:
            images = self.input_norm(images)
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


# ---------------------------------------------------------------------------
# Swin V2 (torchvision backbone loaded via torchgeo)
# ---------------------------------------------------------------------------


class TorchGeoSwinBench(BenchModel):
    """Wrapper for torchgeo Swin-V2 models (swin_v2_b / swin_v2_t).

    These return ``torchvision.models.SwinTransformer`` instances.  We replace
    ``.head`` with ``Identity()`` to get headless features.
    """

    def __init__(
        self,
        num_channels: int,
        factory: str = "swin_v2_b",
        weights_class: str = "Swin_V2_B_Weights",
        weights_member: str = "NAIP_RGB_MI_SATLAS",
        auto_resize: bool = True,
        target_size: int | None = 256,
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_channels=num_channels)
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.backbone = _resolve_torchgeo_factory(factory)(weights=weights)
        self.backbone.head = nn.Identity()
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.input_norm = _extract_normalize_transforms(weights)

    @torch.no_grad()
    def forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return headless Swin-V2 embeddings of shape ``(B, K)``."""
        del bboxes
        if self.input_norm is not None:
            images = self.input_norm(images)
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


# ---------------------------------------------------------------------------
# ScaleMAE (ViT backbone)
# ---------------------------------------------------------------------------


class TorchGeoScaleMAEBench(BenchModel):
    """Wrapper for torchgeo ScaleMAE-Large.

    ``forward_features()`` returns ``(B, N+1, D)`` tokens; we average spatial
    tokens (dropping CLS at index 0) to produce ``(B, D)``.
    """

    def __init__(
        self,
        num_channels: int,
        factory: str = "scalemae_large_patch16",
        weights_class: str = "ScaleMAELarge16_Weights",
        weights_member: str = "FMOW_RGB",
        auto_resize: bool = True,
        target_size: int | None = 224,
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_channels=num_channels)
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.backbone = _resolve_torchgeo_factory(factory)(weights=weights)
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.input_norm = _extract_normalize_transforms(weights)

    @torch.no_grad()
    def forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return mean-pooled spatial tokens of shape ``(B, D)``."""
        del bboxes
        if self.input_norm is not None:
            images = self.input_norm(images)
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        tokens = self.backbone.forward_features(images)  # (B, N+1, D)
        # Average spatial tokens, skip CLS token at index 0
        return tokens[:, 1:, :].mean(dim=1)  # (B, D)


# ---------------------------------------------------------------------------
# DOFA (band-agnostic ViT requiring wavelength input)
# ---------------------------------------------------------------------------


class TorchGeoDOFABench(BenchModel):
    """Wrapper for torchgeo DOFA models (dofa_base / dofa_large).

    DOFA requires a list of wavelengths (one per input channel in µm).
    ``forward_features(x, wavelengths)`` returns ``(B, D)``.
    """

    # Approximate centre wavelengths in µm for Sentinel-2 RGB (B4, B3, B2)
    S2_RGB_WAVELENGTHS: list[float] = [0.665, 0.56, 0.49]

    def __init__(
        self,
        num_channels: int,
        factory: str = "dofa_base_patch16_224",
        weights_class: str = "DOFABase16_Weights",
        weights_member: str = "DOFA_MAE",
        wavelengths: list[float] | None = None,
        auto_resize: bool = True,
        target_size: int | None = 224,
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_channels=num_channels)
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.backbone = _resolve_torchgeo_factory(factory)(weights=weights)
        self.wavelengths = wavelengths or self.S2_RGB_WAVELENGTHS
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.input_norm = _extract_normalize_transforms(weights)

    @torch.no_grad()
    def forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return DOFA feature embeddings of shape ``(B, D)``."""
        del bboxes
        if self.input_norm is not None:
            images = self.input_norm(images)
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone.forward_features(images, wavelengths=self.wavelengths)


# ---------------------------------------------------------------------------
# EarthLoc (place-recognition descriptor)
# ---------------------------------------------------------------------------


class TorchGeoEarthLocBench(BenchModel):
    """Wrapper for torchgeo EarthLoc.

    ``forward(x)`` returns a ``(B, 4096)`` global descriptor.
    """

    def __init__(
        self,
        num_channels: int,
        factory: str = "earthloc",
        weights_class: str = "EarthLoc_Weights",
        weights_member: str = "SENTINEL2_RESNET50",
        auto_resize: bool = True,
        target_size: int | None = 320,
        **_kwargs: Any,
    ) -> None:
        super().__init__(num_channels=num_channels)
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.backbone = _resolve_torchgeo_factory(factory)(weights=weights)
        self.auto_resize = auto_resize
        self.target_size = target_size
        self.input_norm = _extract_normalize_transforms(weights)

    @torch.no_grad()
    def forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return EarthLoc global descriptor of shape ``(B, 4096)``."""
        del bboxes
        if self.input_norm is not None:
            images = self.input_norm(images)
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)
