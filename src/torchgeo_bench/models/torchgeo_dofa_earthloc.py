"""DOFA and EarthLoc torchgeo wrappers.

DOFA is a band-agnostic ViT that requires a per-channel wavelength; EarthLoc
is a ResNet50-based place-recognition descriptor.  Grouped together as two
compact, independently-pretrained backbones rather than split into two
near-empty files.
"""

from typing import Any

import torch

from torchgeo_bench.datasets.base import BandSpec

from ._torchgeo_base import _SAR_SENSORS, _adapt_first_conv, _auto_resize, _TorchGeoBackboneBench

#: DOFA's own pretraining wave table (github.com/zhu-xlab/DOFA,
#: ``pretraining/datasets/waves.json``) assigns both channels of its 2-band
#: Sentinel-1 (VH, VV) modality this placeholder wavelength: ``"2": [3.75,
#: 3.75]``. Radar backscatter has no optical wavelength, so this is a
#: deliberate, sourced placeholder from the original authors -- not a guess
#: -- used to give the wavelength-conditioned hypernetwork a fixed token for
#: "this is the SAR modality" rather than raising.
_DOFA_SAR_WAVELENGTH_UM = 3.75


def _resolve_dofa_wavelengths(
    bands: list[BandSpec],
    wavelengths: list[float] | None,
) -> list[float]:
    """Return one DOFA wavelength per selected input channel.

    Raises on any ``BandSpec`` lacking ``wavelength_um`` whose canonical
    band name has no known fallback, rather than silently defaulting to
    ~green (0.6 µm).  DOFA's wavelength embedding is the only way the model
    "knows" what spectral channel each tensor index represents; a silent
    default would assign green-band weights to e.g. thermal or elevation
    channels and quietly produce garbage features. Two documented
    exceptions: SAR bands (``sensor in {"s1", "sar"}``) get DOFA's own
    placeholder, :data:`_DOFA_SAR_WAVELENGTH_UM`; other optical bands with
    no declared wavelength (e.g. a Landsat dataset that never set one) fall
    back to the true Sentinel-2 centre wavelength for that canonical band
    name via :data:`~torchgeo_bench.models._band_mapping.S2_WAVELENGTHS_UM`.
    Callers that want a different default must pass an explicit
    ``wavelengths=`` list.
    """
    from ._band_mapping import S2_WAVELENGTHS_UM, canonical_band_name

    if wavelengths is not None:
        if len(wavelengths) != len(bands):
            raise ValueError(
                f"DOFA wavelengths length {len(wavelengths)} must match "
                f"selected channel count {len(bands)}."
            )
        return [float(w) for w in wavelengths]

    def _resolved(b: BandSpec) -> float | None:
        if b.wavelength_um is not None:
            return float(b.wavelength_um)
        if b.sensor in _SAR_SENSORS:
            return _DOFA_SAR_WAVELENGTH_UM
        return S2_WAVELENGTHS_UM.get(canonical_band_name(b.name))

    resolved = [_resolved(b) for b in bands]
    missing = [b.name for b, w in zip(bands, resolved, strict=True) if w is None]
    if missing:
        raise ValueError(
            f"DOFA wavelengths missing for {missing}: every BandSpec must have a "
            f"`wavelength_um` set.  SAR / non-optical channels need either an "
            f"explicit wavelength or to be filtered out of the input."
        )
    return [w for w in resolved if w is not None]


class TorchGeoDOFABench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo DOFA models (dofa_base / dofa_large).

    DOFA requires a list of wavelengths (one per input channel in µm).
    ``forward_features(x, wavelengths)`` returns ``(B, D)``.
    """

    # No magnitude check — DOFA's pretrained transform is empty in current
    # torchgeo releases, and dataset units vary widely.
    weights_input_unit = None

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
            **_kwargs,
        )
        self.wavelengths = _resolve_dofa_wavelengths(bands, wavelengths)

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone.forward_features(images, wavelengths=self.wavelengths)


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
            **_kwargs,
        )
        # EarthLoc wraps a ResNet50; adapt its first conv for N-channel input.
        # Results on N!=3 channels are "adapted*" (input-conv weights are
        # timm-averaged, not the pretrained RGB ones).
        _adapt_first_conv(self.backbone, "backbone.conv1", len(bands))

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)
