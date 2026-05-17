"""OlmoEarth model wrapper for torchgeo-bench.

Wraps the OlmoEarth geospatial foundation model (AI2) for use with the
BenchModel interface.  Multi-modal: the wrapper auto-selects OlmoEarth's
``Modality.SENTINEL2_L2A`` / ``LANDSAT`` / ``NAIP`` based on the input
``BandSpec.sensor`` field and builds the right channel layout, band-set
mask, and ``MaskedOlmoEarthSample`` field for each.  This is essential
because OlmoEarth was pretrained with per-modality wavelength
embeddings and band-set structure; feeding Landsat or NAIP data through
the S2 path zero-fills 9 wrong channels and collapses accuracy.

When fewer bands than the target modality provides are available, the
wrapper zero-fills the remaining positions (mask stays all-visible so
``pool_spatially`` works).

Reference implementations (canonical first):
    https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Inference-Quickstart.md
    https://github.com/isaaccorley/geopool/blob/main/scripts/embed_olmoearth.py
"""

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import detect_input_unit, to_s2_dn
from .interface import BenchModel

logger = logging.getLogger(__name__)


# Canonical S2 band order kept as a module constant for backward
# compatibility with tests and external callers.
OLMOEARTH_S2_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
)


# Per-modality layout — kept as a single source of truth so the build
# steps below (channel layout, mask shape, sample field) all agree.
#
# ``name_to_idx`` maps any ``BandSpec.name`` we expect to see in our
# datasets (lower-cased) to the corresponding position in OlmoEarth's
# expected band_order for the modality.  Both semantic names (``blue``,
# ``red_edge_1``) and source-style names (``b02``, ``b04``) are
# accepted so the wrapper works with either GeoBench V1 or V2 datasets.
_MODALITY_INFO: dict[str, dict] = {
    "s2": {
        "modality_name": "SENTINEL2_L2A",
        "sample_field": "sentinel2_l2a",
        "channels": 12,
        "num_band_sets": 3,
        # OlmoEarth band_order: B02, B03, B04, B08, B05, B06, B07, B8A,
        # B11, B12, B01, B09.
        "name_to_idx": {
            "blue": 0, "b02": 0,
            "green": 1, "b03": 1,
            "red": 2, "b04": 2,
            "nir": 3, "b08": 3,
            "red_edge_1": 4, "b05": 4,
            "red_edge_2": 5, "b06": 5,
            "red_edge_3": 6, "b07": 6,
            "red_edge_4": 7, "b8a": 7,
            "swir_1": 8, "b11": 8,
            "swir_2": 9, "b12": 9,
            "coastal_aerosol": 10, "b01": 10,
            "water_vapour": 11, "b09": 11,
        },
    },
    "landsat": {
        "modality_name": "LANDSAT",
        "sample_field": "landsat",
        "channels": 11,
        "num_band_sets": 2,
        # OlmoEarth band_order: B8 (pan), B1, B2, B3, B4, B5, B6, B7,
        # B9, B10, B11.  GeoBench m-forestnet (Landsat-8) typically
        # ships only B2/B3/B4/B5/B6/B7 under semantic names.
        "name_to_idx": {
            "panchromatic": 0, "pan": 0, "b8": 0,
            "coastal_aerosol": 1, "coastal": 1, "b1": 1,
            "blue": 2, "b2": 2,
            "green": 3, "b3": 3,
            "red": 4, "b4": 4,
            "nir": 5, "b5": 5,
            "swir_1": 6, "b6": 6,
            "swir_2": 7, "b7": 7,
            "cirrus": 8, "b9": 8,
            "tirs_1": 9, "thermal_1": 9, "b10": 9,
            "tirs_2": 10, "thermal_2": 10, "b11": 10,
        },
    },
    # NAIP is *not* in olmoearth-pretrain-minimal's supported modalities
    # list (the released encoders only cover S2 / S1 / Landsat / various
    # raster auxiliaries — no NAIP/aerial branch).  So for NAIP / aerial /
    # treesatai-aerial inputs we route the RGB channels through the S2
    # path with the non-RGB S2 positions zero-filled.  This matches the
    # original wrapper behaviour pre-refactor.
    "aerial": {
        "modality_name": "SENTINEL2_L2A",
        "sample_field": "sentinel2_l2a",
        "channels": 12,
        "num_band_sets": 3,
        # S2 positions for the RGB triplet; NIR/IR if present goes to B08.
        "name_to_idx": {
            "red": 2, "r": 2,
            "green": 1, "g": 1,
            "blue": 0, "b": 0,
            "nir": 3, "ir": 3,
        },
    },
}
# Aliases.
_MODALITY_INFO["naip"] = _MODALITY_INFO["aerial"]


def _resolve_modality(bands: list[BandSpec]) -> dict:
    """Pick the OlmoEarth modality from the input ``BandSpec.sensor`` field.

    Raises if the sensor isn't one we have a layout for.
    """
    sensor = bands[0].sensor.lower()
    if not all(b.sensor.lower() == sensor for b in bands):
        sensors = sorted({b.sensor.lower() for b in bands})
        raise ValueError(
            f"OlmoEarth wrapper expects a single sensor per call; "
            f"got mixed sensors {sensors}."
        )
    if sensor not in _MODALITY_INFO:
        raise ValueError(
            f"OlmoEarth wrapper has no layout for sensor '{sensor}'.  "
            f"Supported: {sorted(set(_MODALITY_INFO))}."
        )
    return _MODALITY_INFO[sensor]


class OlmoEarthBenchModel(BenchModel):
    """BenchModel wrapper for OlmoEarth geospatial foundation models.

    OlmoEarth is a multi-modal ViT trained on Sentinel-2, Sentinel-1,
    Landsat, NAIP, and other Earth-observation streams by AI2.  The
    wrapper picks the right modality from ``bands[0].sensor`` and
    constructs a properly-shaped batch for OlmoEarth's encoder.

    Supported modalities (auto-detected):

    * ``"s2"`` -> ``Modality.SENTINEL2_L2A`` (12 channels, 3 band-sets)
    * ``"landsat"`` -> ``Modality.LANDSAT`` (11 channels, 2 band-sets)
    * ``"aerial"`` / ``"naip"`` -> ``Modality.NAIP`` (4 channels, 1 band-set)

    Channels missing from the input are zero-filled at the corresponding
    OlmoEarth position; the mask stays all-visible so ``pool_spatially``
    can still produce embeddings.

    The wrapper overrides ``normalize_inputs`` to identity — OlmoEarth's
    internal ``Normalizer`` consumes raw values directly.  Input scale
    (DN / reflectance / uint8) is auto-detected and rescaled to S2 DN
    before normalisation.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        model_size: One of ``"nano"``, ``"tiny"``, ``"base"``, ``"large"``.
        patch_size: Patch size for the encoder (default 4 per the AI2
            quickstart; smaller is usually better but uses more GPU memory).
        input_res: Input resolution in meters (default 10, S2 GSD).
        time_steps: Temporal slots in the input.  Default 1 — the AI2
            quickstart's canonical single-image setting.
        std_multiplier: Std multiplier passed to ``Normalizer``.
        normalize: If True, L2-normalize output embeddings.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        model_size: Literal["nano", "tiny", "base", "large"] = "base",
        patch_size: int = 4,
        input_res: int = 10,
        time_steps: int = 1,
        std_multiplier: float = 2.0,
        normalize: bool = False,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)

        # Lazy imports so the package is only needed when this model is used.
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

        # Quiet down the package's INFO-level ModalitySpec dumps.
        for logger_name in (
            "olmoearth_pretrain_minimal",
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1",
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        # Resolve the modality layout from the bands' sensor.
        info = _resolve_modality(self.bands)
        self._modality_info = info
        self._modality = getattr(Modality, info["modality_name"])
        self._sample_field = info["sample_field"]
        self._target_channels = info["channels"]
        self._num_band_sets = info["num_band_sets"]

        # Map each input band position to its slot in OlmoEarth's modality
        # band_order.  Unknown band names raise — quiet zero-fill on every
        # channel would degrade silently.
        name_to_idx = info["name_to_idx"]
        self._band_indices: list[int] = []
        unknown: list[str] = []
        for b in self.bands:
            key = b.name.lower()
            if key not in name_to_idx:
                unknown.append(b.name)
            else:
                self._band_indices.append(name_to_idx[key])
        if unknown:
            raise ValueError(
                f"OlmoEarth wrapper can't map BandSpec names {unknown} for "
                f"sensor '{self.bands[0].sensor}'.  Add them to "
                f"_MODALITY_INFO['{self.bands[0].sensor.lower()}']['name_to_idx'] "
                f"with the correct OlmoEarth band index."
            )

        self.model_size = model_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.time_steps = time_steps
        self.do_normalize = normalize

        # OlmoEarth's internal Normalizer expects raw S2 DN (0..~10000).
        # Datasets vary: DN, reflectance [0, 2.8], uint8 [0, 255].  Detect
        # once and rescale per batch in _forward_patch_features.
        self._input_unit = detect_input_unit(self.bands)

        model_id = getattr(ModelID, f"OLMOEARTH_V1_{model_size.upper()}")
        self.encoder_model = load_model_from_id(model_id, load_weights=True)
        self.normalizer = Normalizer(std_multiplier=std_multiplier)

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Identity — OlmoEarth's internal Normalizer handles raw values."""
        return images

    def _pad_to_modality_layout(self, images: torch.Tensor) -> torch.Tensor:
        """Place each input channel at its OlmoEarth modality position.

        Returns a ``(B, target_channels, H, W)`` tensor with zeros wherever
        the input doesn't supply a band.  Operates on the GPU.
        """
        B, _, H, W = images.shape
        out = torch.zeros(
            B, self._target_channels, H, W, device=images.device, dtype=images.dtype
        )
        for src_idx, dst_idx in enumerate(self._band_indices):
            out[:, dst_idx] = images[:, src_idx]
        return out

    def _build_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Per-band-set ``(B, H, W, T, num_band_sets)`` mask — all visible.

        ``MaskValue.ONLINE_ENCODER`` is 0 so zeros == visible.  Shape must
        be 5-D because the encoder's ``apply_embedding_to_modality`` does
        ``mask[..., idx]`` for each band-set index.
        """
        return torch.zeros(
            B, H, W, self.time_steps, self._num_band_sets, dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract image-level embeddings from raw inputs."""
        del bboxes
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        device = images.device

        # Rescale to S2 DN before any further processing.  No-op when the
        # dataset already delivers DN.  (We use S2 DN as the canonical
        # scale even for Landsat / NAIP — the modality-specific Normalizer
        # then re-centres on per-band statistics.)
        images = to_s2_dn(images, self._input_unit)

        # Map input channels -> OlmoEarth modality layout.
        images = self._pad_to_modality_layout(images)

        B, C, H, W = images.shape

        images_nhwc = images.permute(0, 2, 3, 1).cpu().numpy()
        images_nhwtc = images_nhwc[:, :, :, None, :]
        if self.time_steps > 1:
            images_nhwtc = np.repeat(images_nhwtc, self.time_steps, axis=3)
        images_nhwtc = self.normalizer.normalize(self._modality, images_nhwtc)

        timestamps = torch.zeros(B, self.time_steps, 3, dtype=torch.long, device=device)
        timestamps[:, :, 0] = 15
        timestamps[:, :, 1] = 6
        timestamps[:, :, 2] = 2020

        modality_mask = self._build_mask(B, H, W, device)

        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            **{
                self._sample_field: torch.from_numpy(images_nhwtc).float().to(device),
                f"{self._sample_field}_mask": modality_mask,
            },
        )

        outputs = self.encoder_model.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=True,
        )

        pooled = outputs["tokens_and_masks"].pool_spatially(PoolingType.MEAN)
        embeddings = pooled.mean(dim=(1, 2))

        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
