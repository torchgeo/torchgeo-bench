"""TerraTorch backbone wrappers for torchgeo-bench."""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from ._band_mapping import map_to_model_bands
from ._input_units import InputUnit
from .interface import BenchModel

logger = logging.getLogger(__name__)


def _build_backbone(name: str, **kwargs: Any) -> nn.Module:
    try:
        import terratorch.models.backbones  # noqa: F401 — populate registry
        from terratorch.registry import BACKBONE_REGISTRY
    except ImportError as e:
        raise ImportError(
            "terratorch is required for this wrapper; install with "
            "`pip install torchgeo-bench[terratorch]`."
        ) from e
    return BACKBONE_REGISTRY.build(name, **kwargs)


def _maybe_resize(images: torch.Tensor, size: int | None) -> torch.Tensor:
    if size is None:
        return images
    h, w = images.shape[-2:]
    if h == size and w == size:
        return images
    return F.interpolate(images, size=(size, size), mode="bicubic", align_corners=False)


def _pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool ``(B, N, D)`` tokens, dropping a CLS token if N is square+1."""
    if tokens.ndim != 3:
        return tokens
    n = tokens.shape[1]
    side = int(round(n**0.5))
    if side * side == n - 1:
        return tokens[:, 1:, :].mean(dim=1)
    return tokens.mean(dim=1)


def _reduce_to_vec(out: torch.Tensor | list | tuple) -> torch.Tensor:
    if isinstance(out, list | tuple):
        out = out[-1]
    if out.ndim == 4:
        return out.mean(dim=(-2, -1))
    if out.ndim == 3:
        return _pool_tokens(out)
    return out


class _TerraTorchBench(BenchModel):
    """Shared scaffold for TerraTorch backbone wrappers."""

    backbone_name: str = ""

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        target_size: int | None = 224,
        backbone_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(bands=bands, **kwargs)
        self.target_size = target_size
        self.backbone = _build_backbone(self.backbone_name, **(backbone_kwargs or {}))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        return images

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del bboxes
        x = _maybe_resize(self._prepare_input(images), self.target_size)
        return _reduce_to_vec(self.backbone(x))


PRITHVI_BANDS: list[str] = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]


_PRITHVI_V1_MEAN = [775.0, 1081.0, 1229.0, 2497.0, 2204.0, 1611.0]
_PRITHVI_V1_STD = [1282.0, 1270.0, 1399.0, 1368.0, 1292.0, 1155.0]
_PRITHVI_V2_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
_PRITHVI_V2_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]


class TerraTorchPrithviBench(_TerraTorchBench):
    """IBM/NASA Prithvi-EO v1/v2 — auto-maps dataset bands onto 6 HLS slots @ 224."""

    expected_input_unit = InputUnit.S2_DN

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "prithvi_eo_v2_300",
        pretrained: bool = True,
        target_size: int | None = 224,
        **kwargs: Any,
    ) -> None:
        self.backbone_name = backbone_name
        # `model_native` normalisation needs the right mean/std for this version.
        self.pretrain_mean = _PRITHVI_V1_MEAN if "v1" in backbone_name else _PRITHVI_V2_MEAN
        self.pretrain_std = _PRITHVI_V1_STD if "v1" in backbone_name else _PRITHVI_V2_STD
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs={"pretrained": pretrained, "num_frames": 1},
            **kwargs,
        )

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        mapped, _ = map_to_model_bands(images, self.bands, PRITHVI_BANDS)
        return mapped


CLAY_BANDS: list[str] = ["blue", "green", "red", "nir", "swir1", "swir2"]
_CLAY_WAVELENGTHS_UM: list[float] = [0.493, 0.560, 0.665, 0.842, 1.610, 2.190]


class TerraTorchClayBench(_TerraTorchBench):
    """Clay v1.5 — 6 S2 bands @ 256, conditioned on per-band ``waves`` (µm) and ``gsd``."""

    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "timm_clay_v1_base",
        pretrained: bool = True,
        target_size: int | None = 256,
        modality: str = "sentinel-2-l2a",  # noqa: ARG002 — kept for config back-compat
        gsd: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self.backbone_name = backbone_name
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs={"pretrained": pretrained},
            **kwargs,
        )
        self.gsd = gsd
        self.register_buffer("_clay_waves", torch.tensor(_CLAY_WAVELENGTHS_UM, dtype=torch.float32))

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        mapped, _ = map_to_model_bands(images, self.bands, CLAY_BANDS)
        return mapped

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del bboxes
        x = _maybe_resize(self._prepare_input(images), self.target_size)
        return _reduce_to_vec(self.backbone(x, waves=self._clay_waves.to(x.device), gsd=self.gsd))


TERRAMIND_S2L2A_BANDS: list[str] = [
    "coastal",
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "watervapor",
    "swir1",
    "swir2",
]


class TerraTorchTerraMindBench(_TerraTorchBench):
    """TerraMind v1 — takes ``{modality: (B, 12, H, W)}`` for fixed-channel S2L2A tokenizer."""

    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        backbone_name: str = "terramind_v1_base",
        pretrained: bool = True,
        target_size: int | None = 224,
        modality: str = "S2L2A",
        **kwargs: Any,
    ) -> None:
        self.backbone_name = backbone_name
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs={"pretrained": pretrained, "modalities": [modality]},
            **kwargs,
        )
        self.modality = modality

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del bboxes
        x, _ = map_to_model_bands(images, self.bands, TERRAMIND_S2L2A_BANDS)
        x = _maybe_resize(x, self.target_size)
        return _reduce_to_vec(self.backbone({self.modality: x}))
