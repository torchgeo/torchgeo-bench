"""CROMA and Panopticon torchgeo wrappers (multi-modal ViT backbones)."""

from typing import Any

import torch

from torchgeo_bench.datasets.base import BandSpec

from ._band_mapping import S2_WAVELENGTHS_UM, canonical_band_name, map_to_model_bands
from ._input_units import InputUnit
from ._normalization import NormalizationStrategy
from ._torchgeo_base import _SAR_SENSORS, _auto_resize, _TorchGeoBackboneBench

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
    """CROMA optical-only path: feeds ``s2_encoder`` directly and pools via ``s2_GAP_FFN``.

    CROMA's published preprocessing (github.com/antofuller/CROMA, taken from
    SatMAE/SeCo) is not fixed pretraining constants: it clips each channel to
    ``mean +/- std_multiplier * std`` and rescales that window to ``[0, 1]``.
    ``model_native`` reproduces that using the dataset's own BandSpec stats,
    which is what the authors compute per dataset.
    """

    weights_input_unit = "reflectance_0_1"
    expected_input_unit = InputUnit.REFLECTANCE_0_1

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
        std_multiplier: float = 2.0,
        **_kwargs: Any,
    ) -> None:
        self.std_multiplier = std_multiplier
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
            **_kwargs,
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply CROMA's own preprocessing under model_native.

        Other strategies fall through to the shared implementation, so the
        normalisation ablation still varies for this model.
        """
        if self.normalization is not NormalizationStrategy.MODEL_NATIVE:
            return super().normalize_inputs(images)
        n = images.shape[1]
        mean = torch.tensor([b.mean for b in self.bands], dtype=torch.float32).view(1, n, 1, 1)
        std = torch.tensor([b.std for b in self.bands], dtype=torch.float32).view(1, n, 1, 1)
        lo = (mean - self.std_multiplier * std).to(images.device, images.dtype)
        hi = (mean + self.std_multiplier * std).to(images.device, images.dtype)
        return ((images - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0)

    def _model_native_weights_normalizer(self):
        """CROMA always builds model_native from BandSpec stats in normalize_inputs.

        Never falls back to the generic pretrain_mean/pretrain_std builder
        (CROMA declares neither), regardless of whether the loaded weights
        ship their own ``Normalize`` transform.
        """
        return self._weights_normalize if self._weights_normalize is not None else (lambda x: x)

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        # Bypass CROMA.forward — its joint branch references `sar_encodings`
        # even when only the optical modality is provided.
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        x_opt, _ = map_to_model_bands(images, self.bands, _CROMA_S2_12)
        encodings = self.backbone.s2_encoder(imgs=x_opt, attn_bias=self.backbone.attn_bias)
        return self.backbone.s2_GAP_FFN(encodings.mean(dim=1))


# Polarization -> Panopticon's negative SAR channel-id convention (orbit-
# agnostic group; torchgeo-bench's BandSpecs carry no per-sample orbit
# direction). See github.com/Panopticon-FM/panopticon/blob/main/dinov2/
# configs/data/satellites/sentinel1.yaml.
_PANOPTICON_SAR_MU: dict[str, float] = {"vv": -1.0, "vh": -2.0, "hh": -3.0, "hv": -4.0}


def _resolve_panopticon_chn_ids(bands: list[BandSpec]) -> list[float]:
    """Return one Panopticon ``chn_id`` (nm wavelength, or negative SAR code) per band.

    Panopticon's ``chn_ids`` encode optical wavelength in nm for optical
    channels and a negative integer polarization code for SAR channels (see
    :data:`_PANOPTICON_SAR_MU`), per its own forward-pass docstring. Optical
    bands with no declared ``wavelength_um`` fall back to the true Sentinel-2
    centre wavelength for that canonical band name, mirroring DOFA's fallback
    (:func:`~torchgeo_bench.models.torchgeo_dofa_earthloc._resolve_dofa_wavelengths`).
    """

    def _resolved(b: BandSpec) -> float | None:
        if b.sensor in _SAR_SENSORS:
            return _PANOPTICON_SAR_MU.get(canonical_band_name(b.name))
        if b.wavelength_um is not None:
            return float(b.wavelength_um) * 1000.0
        wl = S2_WAVELENGTHS_UM.get(canonical_band_name(b.name))
        return wl * 1000.0 if wl is not None else None

    resolved = [_resolved(b) for b in bands]
    missing = [b.name for b, w in zip(bands, resolved, strict=True) if w is None]
    if missing:
        raise ValueError(
            f"Panopticon chn_ids missing for {missing}: SAR bands must be one of "
            f"{sorted(_PANOPTICON_SAR_MU)}, and optical bands need a `wavelength_um` "
            f"or a known Sentinel-2 canonical name."
        )
    return [w for w in resolved if w is not None]


class TorchGeoPanopticonBench(_TorchGeoBackboneBench):
    """Panopticon ViT-B/14 — per-channel wavelength tokens (nm) from BandSpec."""

    weights_input_unit = "reflectance_0_1"
    expected_input_unit = InputUnit.REFLECTANCE_0_1

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
            **_kwargs,
        )
        chn_ids = _resolve_panopticon_chn_ids(bands)
        self.register_buffer("_chn_ids", torch.tensor(chn_ids, dtype=torch.float32))

    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        chn_ids = self._chn_ids.unsqueeze(0).expand(images.shape[0], -1)
        if images.requires_grad:
            chn_ids = chn_ids.clone().requires_grad_(True)
        return self.backbone({"imgs": images, "chn_ids": chn_ids})
