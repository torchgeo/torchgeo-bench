"""torchgeo foundation-model wrappers for torchgeo-bench.

Each wrapper class loads a torchgeo pretrained model and exposes the
``BenchModel`` interface.  Inputs are raw sensor values; the wrapper's
``normalize_inputs`` override applies the ``Normalize`` layer attached to
the pretrained weights.

Caveats
-------

The pretrained weights' ``Normalize`` transform was calibrated for a
specific input scale (e.g. Sentinel-2 DN / 10000, NAIP uint8 / 255).
Pairing one of these wrappers with a dataset whose raw values are in a
different scale will silently misnormalize.  Each wrapper sets
:attr:`weights_input_unit` documenting the expected scale, and
:func:`_warn_unit_mismatch` emits a warning when the band statistics
look incompatible.  See GitHub issue
`#16 <https://github.com/torchgeo/torchgeo-bench/issues/16>`_ for the
follow-up on stronger guards.
"""

import logging
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeo.models as tgm
from torchvision.transforms import Normalize as NormalizeV1
from torchvision.transforms.v2 import Normalize as NormalizeV2

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

logger = logging.getLogger(__name__)


def _resolve_torchgeo_factory(factory_name: str):
    """Return the model-factory function from ``torchgeo.models``."""
    fn = getattr(tgm, factory_name, None)
    if fn is None:
        raise ValueError(f"torchgeo.models has no factory function '{factory_name}'")
    return fn


def _resolve_torchgeo_weights(weights_class_name: str, weights_member: str):
    """Return the concrete weights enum member."""
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
    """Extract only the ``Normalize`` layers from a torchgeo weights transform."""
    if not hasattr(weights, "transforms") or weights.transforms is None:
        return None
    transform = weights.transforms
    if callable(transform) and not isinstance(transform, nn.Module):
        transform = transform()
    if isinstance(transform, nn.Identity):
        return None
    try:
        iterator = iter(transform)
    except TypeError:
        return None
    norms = [t for t in iterator if isinstance(t, (NormalizeV1, NormalizeV2))]
    if not norms:
        return None
    return nn.Sequential(*norms)


# Magnitude buckets for `weights_input_unit` plausibility checks.  Keys are
# rough expected per-band mean ranges in raw units.
_UNIT_EXPECTED_MEAN: dict[str, tuple[float, float]] = {
    "uint8_div255": (0.0, 255.0),
    "reflectance_0_1": (0.0, 2.0),
    "s2_dn_div10000": (0.0, 10000.0),
}


def _warn_unit_mismatch(
    cls_name: str,
    weights_input_unit: str | None,
    bands: list[BandSpec],
    check: str,
) -> None:
    """Emit a warning if the per-band ``mean`` magnitude looks incompatible.

    Args:
        cls_name: Wrapper class name, used in the warning message.
        weights_input_unit: Expected input scale tag (key into
            :data:`_UNIT_EXPECTED_MEAN`).  ``None`` skips the check.
        bands: The dataset's :class:`BandSpec` list.
        check: ``"warn"`` (default) emits a UserWarning; ``"error"`` raises;
            ``"ignore"`` is silent.
    """
    if check == "ignore" or weights_input_unit is None:
        return
    expected = _UNIT_EXPECTED_MEAN.get(weights_input_unit)
    if expected is None:
        return
    lo, hi = expected
    bad = [b for b in bands if not (lo <= b.mean <= hi * 1.5)]
    if not bad:
        return
    msg = (
        f"{cls_name}: pretrained weights expect inputs in unit "
        f"{weights_input_unit!r} (per-band mean ~ [{lo}, {hi}]), but the "
        f"selected dataset has bands with mean outside that range: "
        f"{[(b.name, b.mean) for b in bad[:5]]}{'...' if len(bad) > 5 else ''}. "
        "Embeddings may be poorly scaled."
    )
    if check == "error":
        raise RuntimeError(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


class _TorchGeoBackboneBench(BenchModel):
    """Shared scaffolding for torchgeo pretrained-weights wrappers.

    Subclasses set :attr:`weights_input_unit` and implement
    :meth:`_load_backbone` returning the headless ``nn.Module`` to call
    on the normalized input.
    """

    weights_input_unit: str | None = None

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str,
        weights_class: str,
        weights_member: str,
        auto_resize: bool,
        target_size: int | None,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(bands=bands)
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.weights = weights
        self.backbone = self._load_backbone(weights, factory)
        self.auto_resize = auto_resize
        self.target_size = target_size
        self._weights_normalize = _extract_normalize_transforms(weights)
        if input_unit_check not in ("warn", "ignore", "error"):
            raise ValueError(
                f"input_unit_check must be one of warn|ignore|error, got {input_unit_check!r}."
            )
        _warn_unit_mismatch(type(self).__name__, self.weights_input_unit, bands, input_unit_check)

    def _load_backbone(self, weights, factory: str) -> nn.Module:
        return _resolve_torchgeo_factory(factory)(weights=weights)

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the pretrained weights' ``Normalize`` layer (no extra z-score)."""
        if self._weights_normalize is None:
            return images
        return self._weights_normalize(images)


# ---------------------------------------------------------------------------
# ResNet (timm backbone loaded via torchgeo)
# ---------------------------------------------------------------------------


class TorchGeoResNetBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo ResNet models (resnet18 / resnet50 / resnet152).

    These return ``timm.models.resnet.ResNet`` instances.  We replace ``.fc``
    with ``Identity()`` to get headless ``(B, K)`` feature vectors.

    Defaults match the SeCo / MoCo Sentinel-2 RGB pretrained weights, whose
    ``Normalize`` transform expects raw Sentinel-2 DN values divided into
    a single global scale.
    """

    weights_input_unit = "s2_dn_div10000"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "resnet50",
        weights_class: str = "ResNet50_Weights",
        weights_member: str = "SENTINEL2_RGB_MOCO",
        auto_resize: bool = False,
        target_size: int | None = 224,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )
        self.backbone.fc = nn.Identity()

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return headless ResNet embeddings of shape ``(B, K)``."""
        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


# ---------------------------------------------------------------------------
# Swin V2 (torchvision backbone loaded via torchgeo)
# ---------------------------------------------------------------------------


class TorchGeoSwinBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo Swin-V2 models (NAIP / Sentinel-2 SatLAS variants)."""

    weights_input_unit = "uint8_div255"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "swin_v2_b",
        weights_class: str = "Swin_V2_B_Weights",
        weights_member: str = "NAIP_RGB_MI_SATLAS",
        auto_resize: bool = True,
        target_size: int | None = 256,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )
        self.backbone.head = nn.Identity()

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return headless Swin-V2 embeddings of shape ``(B, K)``."""
        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


# ---------------------------------------------------------------------------
# ScaleMAE (ViT backbone)
# ---------------------------------------------------------------------------


class TorchGeoScaleMAEBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo ScaleMAE-Large.

    ``forward_features()`` returns ``(B, N+1, D)`` tokens; we average spatial
    tokens (dropping CLS at index 0) to produce ``(B, D)``.
    """

    weights_input_unit = "uint8_div255"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "scalemae_large_patch16",
        weights_class: str = "ScaleMAELarge16_Weights",
        weights_member: str = "FMOW_RGB",
        auto_resize: bool = True,
        target_size: int | None = 224,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return mean-pooled spatial tokens of shape ``(B, D)``."""
        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        tokens = self.backbone.forward_features(images)  # (B, N+1, D)
        return tokens[:, 1:, :].mean(dim=1)


# ---------------------------------------------------------------------------
# DOFA (band-agnostic ViT requiring wavelength input)
# ---------------------------------------------------------------------------


class TorchGeoDOFABench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo DOFA models (dofa_base / dofa_large).

    DOFA requires a list of wavelengths (one per input channel in µm).
    ``forward_features(x, wavelengths)`` returns ``(B, D)``.

    .. note::

       The wavelength list is currently hard-coded to Sentinel-2 RGB
       (``[0.665, 0.56, 0.49]``).  Plumbing per-band wavelengths from
       :attr:`BenchModel.bands` is tracked in
       `#15 <https://github.com/torchgeo/torchgeo-bench/issues/15>`_.
    """

    # No magnitude check — DOFA's pretrained transform is empty in current
    # torchgeo releases, and dataset units vary widely.
    weights_input_unit = None

    # Approximate centre wavelengths in µm for Sentinel-2 RGB (B4, B3, B2).
    S2_RGB_WAVELENGTHS: list[float] = [0.665, 0.56, 0.49]

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "dofa_base_patch16_224",
        weights_class: str = "DOFABase16_Weights",
        weights_member: str = "DOFA_MAE",
        wavelengths: list[float] | None = None,
        auto_resize: bool = True,
        target_size: int | None = 224,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )
        self.wavelengths = wavelengths or self.S2_RGB_WAVELENGTHS

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return DOFA feature embeddings of shape ``(B, D)``."""
        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone.forward_features(images, wavelengths=self.wavelengths)


# ---------------------------------------------------------------------------
# EarthLoc (place-recognition descriptor)
# ---------------------------------------------------------------------------


class TorchGeoEarthLocBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo EarthLoc.

    ``forward(x)`` returns a ``(B, 4096)`` global descriptor.
    """

    weights_input_unit = "uint8_div255"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "earthloc",
        weights_class: str = "EarthLoc_Weights",
        weights_member: str = "SENTINEL2_RESNET50",
        auto_resize: bool = True,
        target_size: int | None = 320,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return EarthLoc global descriptor of shape ``(B, 4096)``."""
        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


_CROMA_S2_12 = [
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


class TorchGeoCromaBench(_TorchGeoBackboneBench):
    """CROMA optical-only path: feeds ``s2_encoder`` directly and pools via ``s2_GAP_FFN``."""

    weights_input_unit = "s2_dn_div10000"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "croma_base",
        weights_class: str = "CROMABase_Weights",
        weights_member: str = "CROMA_VIT",
        auto_resize: bool = True,
        target_size: int | None = 120,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Bypass CROMA.forward — its joint branch references `sar_encodings`
        # even when only the optical modality is provided.
        from ._band_mapping import map_to_model_bands

        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        x_opt, _ = map_to_model_bands(images, self.bands, _CROMA_S2_12)
        encodings = self.backbone.s2_encoder(imgs=x_opt, attn_bias=self.backbone.attn_bias)
        return self.backbone.s2_GAP_FFN(encodings.mean(dim=1))


class TorchGeoPanopticonBench(_TorchGeoBackboneBench):
    """Panopticon ViT-B/14 — per-channel wavelength tokens (nm) from BandSpec."""

    weights_input_unit = "s2_dn_div10000"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "panopticon_vitb14",
        weights_class: str = "Panopticon_Weights",
        weights_member: str = "VIT_BASE14",
        auto_resize: bool = True,
        target_size: int | None = 224,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
        )
        from ._band_mapping import wavelengths_um

        wls_nm = [w * 1000.0 for w in wavelengths_um(bands, default_um=0.6)]
        self.register_buffer("_chn_ids", torch.tensor(wls_nm, dtype=torch.float32))

    @torch.no_grad()
    def _forward_patch_features(
        self, images: torch.Tensor, bboxes: torch.Tensor | None = None
    ) -> torch.Tensor:
        del bboxes
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        chn_ids = self._chn_ids.unsqueeze(0).expand(images.shape[0], -1)
        return self.backbone({"imgs": images, "chn_ids": chn_ids})
