"""DEO torchgeo wrapper (frozen Swin-B, native RGB or Sentinel-2 inputs)."""

from typing import Any

import torch

from torchgeo_bench.datasets.base import BandSpec

from ._band_mapping import canonical_band_name, map_to_model_bands, resolve_src_indices
from ._input_units import InputUnit, detect_input_unit, to_reflectance, to_s2_dn
from ._normalization import NormalizationStrategy
from ._torchgeo_base import _auto_resize, _resolve_torchgeo_factory, _resolve_torchgeo_weights
from .interface import BenchModel

_DEO_TARGET_SIZE = 224

_DEO_RGB_BANDS = ["red", "green", "blue"]
_DEO_S2_BANDS = [
    "red",
    "green",
    "blue",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir_narrow",
    "swir1",
    "swir2",
]
_DEO_RGB_MEAN = (0.485, 0.456, 0.406)
_DEO_RGB_STD = (0.229, 0.224, 0.225)
_DEO_S2_MEAN = (
    0.4182007312774658,
    0.4214799106121063,
    0.3991275727748871,
    1263.73947144,
    1645.40315151,
    1846.87040806,
    1762.59530783,
    1972.62420416,
    1732.16362238,
    1247.91870117,
)
_DEO_S2_STD = (
    0.28774282336235046,
    0.27541765570640564,
    0.2764017581939697,
    948.9819932,
    1108.06650639,
    1258.36394548,
    1233.1492281,
    1364.38688993,
    1310.36996126,
    1087.6020813,
)


class TorchGeoDEOBench(BenchModel):
    """Frozen DEO Swin-B wrapper for native RGB or Sentinel-2 inputs.

    DEO has separate RGB and ten-channel Sentinel-2 patch embeddings. Its
    normalization deliberately bypasses the generic TorchGeo weights transform:
    RGB uses ImageNet statistics, while S2 uses DEO's exact published values.
    """

    expected_input_unit = InputUnit.REFLECTANCE_0_1

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        mode: str = "rgb",
        normalization: NormalizationStrategy | str = NormalizationStrategy.MODEL_NATIVE,
        auto_resize: bool = True,
        target_size: int = _DEO_TARGET_SIZE,
        **_kwargs: Any,
    ) -> None:
        if NormalizationStrategy(normalization) is not NormalizationStrategy.MODEL_NATIVE:
            raise ValueError(
                "TorchGeoDEOBench requires normalization='model_native' to preserve "
                "the DEO pretraining input pipeline."
            )
        if mode not in {"rgb", "s2"}:
            raise ValueError("TorchGeoDEOBench mode must be 'rgb' or 's2'.")
        if target_size != _DEO_TARGET_SIZE:
            raise ValueError(f"TorchGeoDEOBench requires target_size={_DEO_TARGET_SIZE}.")

        # DEO does its own native conversion below. Avoid the generic model_native
        # builder, which cannot infer a single unit from mixed sensor inputs.
        super().__init__(bands=bands, normalization=NormalizationStrategy.IDENTITY)
        self.normalization = NormalizationStrategy.MODEL_NATIVE
        self.mode = mode
        self.auto_resize = auto_resize
        self.target_size = target_size

        if mode == "rgb":
            actual = [canonical_band_name(band.name) for band in bands]
            if actual != _DEO_RGB_BANDS:
                raise ValueError(
                    "TorchGeoDEOBench RGB mode requires exactly ordered red, green, blue bands; "
                    f"got {actual}."
                )
            self._rgb_input_unit = detect_input_unit(bands)
            mean, std = _DEO_RGB_MEAN, _DEO_RGB_STD
        else:
            source_indices = resolve_src_indices(bands, preferred_sensors=("s2",))
            missing = [band for band in _DEO_S2_BANDS if band not in source_indices]
            if missing:
                available = sorted(source_indices)
                raise ValueError(
                    "TorchGeoDEOBench S2 mode is missing required Sentinel-2 semantics "
                    f"{missing}; available={available}."
                )
            self._s2_source_indices = [source_indices[band] for band in _DEO_S2_BANDS]
            self._s2_input_unit = detect_input_unit(
                [bands[index] for index in self._s2_source_indices]
            )
            mean, std = _DEO_S2_MEAN, _DEO_S2_STD

        self.register_buffer("_deo_mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("_deo_std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))
        weights = _resolve_torchgeo_weights("DEO_Weights", "DEO_SWIN")
        self.backbone = _resolve_torchgeo_factory("deo_base")(weights=weights)
        self.backbone.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "TorchGeoDEOBench":
        """Keep the pretrained DEO backbone frozen in evaluation mode."""
        del mode
        super().train(False)
        self.backbone.eval()
        return self

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply DEO's mode-specific native input conversion and normalization."""
        if images.shape[1] != len(self.bands):
            raise ValueError(
                f"TorchGeoDEOBench received {images.shape[1]} channels for {len(self.bands)} bands."
            )
        if self.mode == "rgb":
            inputs = to_reflectance(images, self._rgb_input_unit).clamp(0.0, 1.0)
        else:
            mapped, _ = map_to_model_bands(
                images,
                self.bands,
                _DEO_S2_BANDS,
                preferred_sensors=("s2",),
            )
            rgb = mapped[:, :3]
            finite_rgb = torch.where(torch.isfinite(rgb), rgb, torch.zeros_like(rgb))
            rgb_max = finite_rgb.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
            rgb = (finite_rgb / rgb_max).clamp(0.0, 1.0)
            multispectral = to_s2_dn(mapped[:, 3:], self._s2_input_unit)
            inputs = torch.cat((rgb, multispectral), dim=1)
        return (inputs - self._deo_mean.to(inputs)) / self._deo_std.to(inputs)

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize:
            images = _auto_resize(images, self.target_size)
        features = self.backbone(images)
        if features.ndim != 4 or features.shape[-1] != 1024:
            raise RuntimeError(
                "DEO backbone must return a final NHWC Swin map with 1024 channels; "
                f"got shape {tuple(features.shape)}."
            )
        return features.mean(dim=(1, 2))
