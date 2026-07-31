"""TerraTorch backbone wrappers for torchgeo-bench."""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from ._band_mapping import map_to_model_bands, select_src_bands
from ._input_units import InputUnit
from ._pooling import VALID_MODES, pool_tokens
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
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def _reduce_to_vec(out: torch.Tensor | list | tuple, pool: str) -> torch.Tensor:
    if isinstance(out, list | tuple):
        out = out[-1]
    if out.ndim == 4:
        # Spatial feature maps: mean / cls both reduce to a GAP; "both" doubles
        # the feature dim by concatenating GAP + max-pooled features.
        gap = out.mean(dim=(-2, -1))
        if pool == "both":
            mp = out.amax(dim=(-2, -1))
            return torch.cat([gap, mp], dim=-1)
        return gap
    if out.ndim == 3:
        return pool_tokens(out, mode=pool)
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
        pool: str = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(bands=bands, **kwargs)
        if pool not in VALID_MODES:
            raise ValueError(f"pool={pool!r} not in {VALID_MODES}")
        self.target_size = target_size
        self.pool = pool
        self.backbone = _build_backbone(self.backbone_name, **(backbone_kwargs or {}))
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        return images

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        x = _maybe_resize(self._prepare_input(images), self.target_size)
        return _reduce_to_vec(self.backbone(x), pool=self.pool)


PRITHVI_BANDS: list[str] = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]


class TerraTorchPrithviBench(_TerraTorchBench):
    """IBM/NASA Prithvi-EO v1/v2 — auto-maps dataset bands onto 6 HLS slots @ 224.

    ``expected_input_unit = S2_DN``: under ``model_native`` the wrapper
    rescales the input to S2 DN scale before band-mapping.  Per-version
    pretraining mean/std are deliberately *not* applied here — they
    correspond to the post-mapped 6-band layout, while strategy
    normalisation runs on the dataset's raw channel count.  The BandSpec
    z-score (default) and minmax strategies compose cleanly with the
    band-mapping zero-fill that follows.
    """

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
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs={"pretrained": pretrained, "num_frames": 1},
            **kwargs,
        )

    def _prepare_input(self, images: torch.Tensor) -> torch.Tensor:
        mapped, _ = map_to_model_bands(images, self.bands, PRITHVI_BANDS, preferred_sensors=("s2",))
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
        mapped, _ = map_to_model_bands(images, self.bands, CLAY_BANDS, preferred_sensors=("s2",))
        return mapped

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        x = _maybe_resize(self._prepare_input(images), self.target_size)
        return _reduce_to_vec(
            self.backbone(x, waves=self._clay_waves.to(x.device), gsd=self.gsd),
            pool=self.pool,
        )


# Canonical short name -> TerraTorch pretrained band token, in each
# modality's pretrained channel order (terramind_register.PRETRAINED_BANDS).
TERRAMIND_MODALITY_BANDS: dict[str, dict[str, str]] = {
    "S2L2A": {
        "coastal": "COASTAL_AEROSOL",
        "blue": "BLUE",
        "green": "GREEN",
        "red": "RED",
        "rededge1": "RED_EDGE_1",
        "rededge2": "RED_EDGE_2",
        "rededge3": "RED_EDGE_3",
        "nir": "NIR_BROAD",
        "nir_narrow": "NIR_NARROW",
        "watervapor": "WATER_VAPOR",
        "swir1": "SWIR_1",
        "swir2": "SWIR_2",
    },
    "RGB": {"red": "RED", "green": "GREEN", "blue": "BLUE"},
}

TERRAMIND_S2L2A_BANDS: list[str] = list(TERRAMIND_MODALITY_BANDS["S2L2A"])


class TerraTorchTerraMindBench(_TerraTorchBench):
    """TerraMind v1 — native modality input, ``{modality: (B, C, H, W)}``.

    Dataset channels are matched to the modality's pretrained band layout by
    canonical name (Sentinel-2 bands preferred when a name is ambiguous
    across sensors, e.g. TreeSatAI aerial vs S2 ``red``).  When only a
    subset of the pretrained bands is available, the backbone is built with
    TerraTorch's ``bands=`` argument, which selects the matching pretrained
    patch-embedding channels instead of zero-filling absent bands.  RGB-only
    datasets should use ``modality: RGB`` (the dedicated ``untok_sen2rgb``
    adapter), not an RGB subset of ``S2L2A``.
    """

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
        modality_bands = TERRAMIND_MODALITY_BANDS.get(modality)
        if modality_bands is None:
            raise ValueError(
                f"Unsupported TerraMind modality {modality!r}; "
                f"expected one of {sorted(TERRAMIND_MODALITY_BANDS)}."
            )
        indices, selected = select_src_bands(bands, list(modality_bands), preferred_sensors=("s2",))
        backbone_kwargs: dict[str, Any] = {"pretrained": pretrained, "modalities": [modality]}
        if len(selected) < len(modality_bands):
            backbone_kwargs["bands"] = {modality: [modality_bands[name] for name in selected]}
        self.backbone_name = backbone_name
        super().__init__(
            bands=bands,
            target_size=target_size,
            backbone_kwargs=backbone_kwargs,
            **kwargs,
        )
        self.modality = modality
        self.model_bands = selected
        self._band_indices = indices

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[1] != len(self.bands):
            raise ValueError(
                f"TerraMind: images has {images.shape[1]} channels but "
                f"bands has {len(self.bands)} entries."
            )
        x = images[:, self._band_indices]
        x = _maybe_resize(x, self.target_size)
        return _reduce_to_vec(self.backbone({self.modality: x}), pool=self.pool)
