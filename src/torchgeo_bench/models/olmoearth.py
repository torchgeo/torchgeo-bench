"""OlmoEarth model wrapper for torchgeo-bench.

Wraps the OlmoEarth geospatial foundation model (AI2) for use with the
BenchModel interface.  Multi-modal: the wrapper auto-selects OlmoEarth's
``Modality.SENTINEL2_L2A`` / ``SENTINEL1`` / ``LANDSAT`` / ``NAIP`` based
on the input ``BandSpec.sensor`` field and builds the right channel layout,
band-set mask, and ``MaskedOlmoEarthSample`` field for each modality.

Mixed-sensor inputs (e.g. m-so2sat with Sentinel-2 + SAR) are handled by
splitting bands into per-sensor groups and populating the corresponding
``MaskedOlmoEarthSample`` fields simultaneously.

Reference implementations (canonical first):
    https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Inference-Quickstart.md
    https://github.com/isaaccorley/geopool/blob/main/scripts/embed_olmoearth.py
"""

import logging
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import InputUnit, _detect_band_group_unit, to_s2_dn
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
            "blue": 0,
            "b02": 0,
            "green": 1,
            "b03": 1,
            "red": 2,
            "b04": 2,
            "nir": 3,
            "b08": 3,
            "red_edge_1": 4,
            "b05": 4,
            "red_edge_2": 5,
            "b06": 5,
            "red_edge_3": 6,
            "b07": 6,
            "red_edge_4": 7,
            "b8a": 7,
            "swir_1": 8,
            "b11": 8,
            "swir_2": 9,
            "b12": 9,
            "coastal_aerosol": 10,
            "b01": 10,
            "water_vapour": 11,
            "b09": 11,
            # B10 (cirrus) — OlmoEarth has no cirrus slot; skip gracefully.
            "swir_cirrus": None,
            "b10": None,
        },
        # Impute missing channels from the most spectrally-similar present
        # band (matches helios' per-band imputation; see configs.py imputes).
        # Each (src, dst) is an OlmoEarth channel index: dst is filled with a
        # copy of src when dst is absent from the input and src is present.
        # Wavelengths (um): B01 0.443, B02 0.49, B04 0.665, B05 0.705,
        # B06 0.74, B07 0.783, B08 0.842, B8A 0.865, B09 0.945.
        # GeoBench forestnet ships only B02/B03/B04/B8A/B11/B12.
        "imputes": [
            (7, 3),  # B08 NIR        <- B8A (0.842 -> 0.865)
            (2, 4),  # B05 RedEdge1   <- B04 red (0.705 -> 0.665)
            (2, 5),  # B06 RedEdge2   <- B04 red (0.74 -> 0.665)
            (7, 6),  # B07 RedEdge3   <- B8A (0.783 -> 0.865)
            (0, 10),  # B01 Coastal   <- B02 blue (0.443 -> 0.49)
            (7, 11),  # B09 WaterVap  <- B8A (0.945 -> 0.865)
        ],
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
            "panchromatic": 0,
            "pan": 0,
            "b8": 0,
            "coastal_aerosol": 1,
            "coastal": 1,
            "b1": 1,
            "blue": 2,
            "b2": 2,
            "green": 3,
            "b3": 3,
            "red": 4,
            "b4": 4,
            "nir": 5,
            "b5": 5,
            "swir_1": 6,
            "b6": 6,
            "swir_2": 7,
            "b7": 7,
            "cirrus": 8,
            "b9": 8,
            "tirs_1": 9,
            "thermal_1": 9,
            "b10": 9,
            "tirs_2": 10,
            "thermal_2": 10,
            "b11": 10,
        },
        # Impute missing channels from the most spectrally-similar present
        # band — mirrors helios m-forestnet exactly (configs.py imputes +
        # the B8->Green band-name conversion). GeoBench m-forestnet ships
        # only B2/B3/B4/B5/B6/B7 (blue/green/red/nir/swir1/swir2).
        # (src, dst) are OlmoEarth LANDSAT channel indices.
        "imputes": [
            (3, 0),  # B8  Panchromatic <- B3 green (helios band-name map)
            (2, 1),  # B1  Coastal      <- B2 blue
            (7, 8),  # B9  Cirrus       <- B7 swir2
            (7, 9),  # B10 TIRS-1       <- B7 swir2
            (7, 10),  # B11 TIRS-2      <- B7 swir2 (helios B11->Tirs1->swir2)
        ],
    },
    # Sentinel-1 SAR: two channels — vv (0) and vh (1).  OlmoEarth
    # BandSet(["vv", "vh"], 16) with is_multitemporal=True.
    # m-so2sat ships 8 SAR-derived bands (real/imag + Lee-filtered
    # components); all are routed to the nearest vv/vh slot.  When
    # multiple bands land on the same slot (e.g. vv_real and vv_lee both
    # map to 0) the later-indexed source band wins — for m-so2sat that
    # means the Lee-filtered imaginary component overwrites, which is
    # reasonable since all variants carry the same polarisation signal.
    "sar": {
        "modality_name": "SENTINEL1",
        "sample_field": "sentinel1",
        "channels": 2,
        "num_band_sets": 1,
        "name_to_idx": {
            "vv": 0,
            "vv_real": 0,
            "vv_imag": 0,
            "vv_lee": 0,
            "vv_lee_real": 0,
            "vv_lee_imag": 0,
            "vh": 1,
            "vh_real": 1,
            "vh_imag": 1,
            "vh_lee": 1,
            "vh_lee_real": 1,
            "vh_lee_imag": 1,
        },
    },
    # Landsat routed through the S2 normalizer.  Wavelengths align well:
    # B-G-R-NIR-SWIR1-SWIR2 ↔ B02-B03-B04-B08-B11-B12.  Use via
    # sensor_remap={"landsat": "landsat_as_s2"}.
    "landsat_as_s2": {
        "modality_name": "SENTINEL2_L2A",
        "sample_field": "sentinel2_l2a",
        "channels": 12,
        "num_band_sets": 3,
        "name_to_idx": {
            "blue": 0,
            "b2": 0,
            "green": 1,
            "b3": 1,
            "red": 2,
            "b4": 2,
            "nir": 3,
            "b5": 5,
            "swir_1": 8,
            "b6": 8,
            "swir_2": 9,
            "b7": 9,
        },
        # Landsat routed through the S2 layout: present positions are B02/
        # B03/B04/B08(nir)/B11/B12; impute the S2-only slots from the nearest
        # present Landsat band. (src, dst) are S2 channel indices.
        "imputes": [
            (2, 4),  # B05 RedEdge1  <- red
            (2, 5),  # B06 RedEdge2  <- red
            (3, 6),  # B07 RedEdge3  <- nir
            (3, 7),  # B8A           <- nir
            (0, 10),  # B01 Coastal  <- blue
            (3, 11),  # B09 WaterVap <- nir
        ],
    },
    # NAIP / aerial: no dedicated OlmoEarth modality, route RGB through
    # the S2 path with non-RGB positions zero-filled.
    "aerial": {
        "modality_name": "SENTINEL2_L2A",
        "sample_field": "sentinel2_l2a",
        "channels": 12,
        "num_band_sets": 3,
        "name_to_idx": {
            "red": 2,
            "r": 2,
            "green": 1,
            "g": 1,
            "blue": 0,
            "b": 0,
            "nir": 3,
            "ir": 3,
        },
    },
}
# Aliases.
_MODALITY_INFO["naip"] = _MODALITY_INFO["aerial"]


# Canonical GSD (meters) per sensor for OlmoEarth's positional encodings.
# Landsat pixels are 30 m; S2/S1 are 10 m.
_SENSOR_INPUT_RES: dict[str, int] = {
    "s2": 10,
    "sar": 10,  # S1 coregistered to S2 10 m grid in OlmoEarth pretraining
    "landsat": 30,
    "aerial": 1,
    "naip": 1,
}

# Sensors whose raw values should NOT be rescaled to S2 DN — pass as-is to
# OlmoEarth's modality-specific normalizer.  SAR values can be large
# (Lee-filtered max ~10 000) and the S1 Normalizer expects the original scale.
# detect_input_unit returns S2_DN for them, making to_s2_dn a no-op anyway,
# but being explicit avoids surprises if the heuristic ever changes.
_PASSTHROUGH_SENSORS: frozenset[str] = frozenset({"sar"})


def _build_sensor_groups(bands: list[BandSpec]) -> list[dict]:
    """Group bands by sensor and resolve per-group modality metadata.

    Returns a list of dicts (one per unique sensor) with keys:
        sensor, modality_name, sample_field, channels, num_band_sets,
        src_indices (indices into the original ``bands`` list),
        dst_indices (target channel positions inside OlmoEarth's layout).
    Preserves the order sensors first appear in ``bands``.
    """
    order: list[str] = []
    grouped: dict[str, list[tuple[int, BandSpec]]] = defaultdict(list)
    for i, b in enumerate(bands):
        key = b.sensor.lower()
        if key not in grouped:
            order.append(key)
        grouped[key].append((i, b))

    result = []
    for sensor in order:
        if sensor not in _MODALITY_INFO:
            supported = sorted(set(_MODALITY_INFO))
            raise ValueError(
                f"OlmoEarth wrapper has no layout for sensor '{sensor}'.  Supported: {supported}."
            )
        info = _MODALITY_INFO[sensor]
        name_to_idx = info["name_to_idx"]
        src_indices: list[int] = []
        dst_indices: list[int] = []
        unknown: list[str] = []
        for src_idx, b in grouped[sensor]:
            key_name = b.name.lower()
            if key_name not in name_to_idx:
                unknown.append(b.name)
            elif name_to_idx[key_name] is not None:
                src_indices.append(src_idx)
                dst_indices.append(name_to_idx[key_name])
            # else: known-but-skippable band (e.g. swir_cirrus / B10 for S2) — zero-filled
        if unknown:
            raise ValueError(
                f"OlmoEarth wrapper can't map BandSpec names {unknown} for "
                f"sensor '{sensor}'.  Add them to "
                f"_MODALITY_INFO['{sensor}']['name_to_idx'] "
                f"with the correct OlmoEarth band index."
            )
        group_bands = [bands[i] for i in src_indices]
        input_unit: InputUnit | None = (
            None if sensor in _PASSTHROUGH_SENSORS else _detect_band_group_unit(group_bands)
        )
        # Resolve which imputations actually apply for this input: fill a
        # missing OlmoEarth channel from the most-similar present band so the
        # encoder never sees fabricated zeros (matches helios). Only fire when
        # the target channel is absent and the source channel is present.
        filled = set(dst_indices)
        impute_ops: list[tuple[int, int]] = []
        for src_dst, tgt_dst in info.get("imputes", []):
            if tgt_dst in filled:
                continue  # real band present — never overwrite it
            if src_dst not in filled:
                logger.warning(
                    "OlmoEarth %s: cannot impute channel %d (source channel %d "
                    "is also missing); leaving it zero-filled.",
                    sensor,
                    tgt_dst,
                    src_dst,
                )
                continue
            impute_ops.append((src_dst, tgt_dst))
        result.append(
            {
                "sensor": sensor,
                "modality_name": info["modality_name"],
                "sample_field": info["sample_field"],
                "channels": info["channels"],
                "num_band_sets": info["num_band_sets"],
                "src_indices": src_indices,
                "dst_indices": dst_indices,
                "input_unit": input_unit,
                "impute_ops": impute_ops,
                # Per-band dataset stats (src order) for the dataset-stats
                # normalization path (norm_from_pretrained=False).
                "src_means": [b.mean for b in group_bands],
                "src_stds": [b.std for b in group_bands],
            }
        )
    return result


def _resolve_modality(bands: list[BandSpec]) -> dict:
    """Pick the OlmoEarth modality from the input ``BandSpec.sensor`` field.

    Single-sensor convenience wrapper around ``_build_sensor_groups``.
    Raises if the sensor isn't one we have a layout for, or if the bands
    span multiple sensors (use ``_build_sensor_groups`` directly instead).
    """
    sensor = bands[0].sensor.lower()
    if not all(b.sensor.lower() == sensor for b in bands):
        sensors = sorted({b.sensor.lower() for b in bands})
        raise ValueError(
            f"OlmoEarth wrapper expects a single sensor per call; got mixed sensors {sensors}."
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
    wrapper picks the right modality (or modalities) from
    ``bands[0].sensor`` and constructs a properly-shaped batch for
    OlmoEarth's encoder.

    Supported modalities (auto-detected from ``BandSpec.sensor``):

    * ``"s2"`` -> ``Modality.SENTINEL2_L2A`` (12 channels, 3 band-sets)
    * ``"landsat"`` -> ``Modality.LANDSAT`` (11 channels, 2 band-sets)
    * ``"sar"`` -> ``Modality.SENTINEL1`` (2 channels, 1 band-set)
    * ``"aerial"`` / ``"naip"`` -> S2 path with RGB zero-fill

    Mixed-sensor inputs (e.g. ``["s2", "sar"]``) are handled by
    building separate tensor branches and populating multiple
    ``MaskedOlmoEarthSample`` fields simultaneously.

    Channels missing from the input are imputed from the most spectrally
    similar band that *is* present (e.g. Landsat cirrus <- swir2), matching
    helios' per-dataset imputation.  Imputation is applied after
    normalization so the imputed channel carries its source band's
    normalized value rather than a fabricated ``(0 - mean) / std`` constant.
    A missing channel with no present source band stays zero-filled.  The
    mask stays all-visible so ``pool_spatially`` can still produce
    embeddings.

    The wrapper overrides ``normalize_inputs`` to identity and normalizes
    internally.  With ``norm_from_pretrained=True`` (default) the input scale
    (DN / reflectance / uint8) is auto-detected per sensor group, rescaled to
    S2 DN, and passed to OlmoEarth's pretrained per-modality ``Normalizer``
    (SAR is passed as-is).  With ``norm_from_pretrained=False`` each band is
    normalized with its own ``BandSpec`` stats instead — use this when the
    input scale can't be matched to the pretraining range (e.g. GeoBench's
    uint8 Landsat).

    ``input_res`` is auto-detected from the primary sensor's GSD: 10 m
    for S2/SAR, 30 m for Landsat.  Pass ``input_res`` explicitly to
    override.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels.
        model_size: One of ``"nano"``, ``"tiny"``, ``"small"``, ``"base"``,
            ``"large"``.  ``"large"`` is only available for ``version="v1"``;
            ``"small"`` is only available for ``version="v1_2"``.
        version: Model version — ``"v1"`` (default), ``"v1_1"`` or
            ``"v1_2"``.  v1.1 ships Nano/Tiny/Base with improved accuracy and
            ~25% more parameters.  v1.2 ships Nano/Tiny/Small/Base (RoPE
            position encoding); no Large variant for v1.1/v1.2.
        patch_size: Patch size for the encoder (default 8).
        input_res: Input resolution in meters.  ``None`` (default) lets
            the wrapper auto-detect from the primary sensor GSD.
        time_steps: Temporal slots in the input.  Default 1 (single
            timestep — the native shape of these classification datasets).
            Values > 1 replicate the single input frame into that many
            identical slots; use only for explicit multi-timestep ablations.
        std_multiplier: Std multiplier passed to ``Normalizer``.
        normalize: If True, L2-normalize output embeddings.
        sar_log_scale: If True, convert SAR values to dB via
            ``10·log10(max(v, 1e-6))`` before feeding OlmoEarth's S1
            normalizer, which was trained on σ⁰ dB values.
        landsat_scale_factor: Optional multiplier applied to Landsat
            values *after* the standard uint8→DN conversion.  Use to
            compensate for mis-matched scales between GeoBench's uint8
            composites and OlmoEarth's pretraining DN range (~10 000).
            Only applies on the pretrained-normalizer path
            (``norm_from_pretrained=True``).
        norm_from_pretrained: If True (default), rescale inputs to S2 DN and
            apply OlmoEarth's pretrained per-modality ``Normalizer`` (correct
            when the input can be matched to the pretraining scale, e.g. S2).
            If False, normalize each band with its own ``BandSpec`` mean/std
            using the same ``±std_multiplier·σ`` no-clip mapping OlmoEarth saw
            in pretraining — i.e. dataset-specific stats.  This is required
            when the input scale doesn't match the pretrained normalizer (e.g.
            GeoBench's uint8 Landsat, where the pretrained Landsat stats assume
            real DN), and matches helios' ``norm_stats_from_pretrained=False`` /
            ``NORM_NO_CLIP_2_STD``.  Supersedes ``landsat_scale_factor`` for
            that case (the DN rescale is skipped entirely).
        sensor_remap: Optional dict mapping sensor names to alternate
            routing keys, e.g. ``{"landsat": "landsat_as_s2"}`` to route
            Landsat bands through the S2 normalizer (+6.6 pp on m-forestnet).
        min_image_size: If set, upsample inputs smaller than this value
            to ``min_image_size × min_image_size`` via bilinear interpolation.
            Useful for datasets with small native images (e.g. m-so2sat at
            32 px) where the patch grid would otherwise be too sparse.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        model_size: Literal["nano", "tiny", "small", "base", "large"] = "base",
        version: Literal["v1", "v1_1", "v1_2"] = "v1",
        patch_size: int = 4,
        input_res: int | None = None,
        time_steps: int = 1,
        std_multiplier: float = 2.0,
        normalize: bool = False,
        sar_log_scale: bool = False,
        landsat_scale_factor: float | None = None,
        norm_from_pretrained: bool = True,
        sensor_remap: dict[str, str] | None = None,
        min_image_size: int | None = None,
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

        # Optionally remap sensor names before routing.
        bands_for_routing = self.bands
        if sensor_remap:
            from dataclasses import replace as dc_replace

            bands_for_routing = [
                dc_replace(b, sensor=sensor_remap.get(b.sensor, b.sensor)) for b in self.bands
            ]

        # Build per-sensor groups (handles both single and mixed sensors).
        sensor_groups = _build_sensor_groups(bands_for_routing)
        for g in sensor_groups:
            g["modality"] = getattr(Modality, g["modality_name"])
        self._sensor_groups = sensor_groups

        # Auto-detect input_res from primary sensor unless explicitly set.
        if input_res is None:
            sensors_present = {g["sensor"] for g in sensor_groups}
            # For mixed s2+sar, S2 10 m is the OlmoEarth pretraining grid.
            primary = "s2" if "s2" in sensors_present else sensor_groups[0]["sensor"]
            input_res = _SENSOR_INPUT_RES.get(primary, 10)

        self.model_size = model_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.time_steps = time_steps
        self.std_multiplier = std_multiplier
        self.do_normalize = normalize
        self.sar_log_scale = sar_log_scale
        self.landsat_scale_factor = landsat_scale_factor
        self.norm_from_pretrained = norm_from_pretrained
        self.min_image_size = min_image_size

        model_id = getattr(ModelID, f"OLMOEARTH_{version.upper()}_{model_size.upper()}")
        self.encoder_model = load_model_from_id(model_id, load_weights=True)
        self.normalizer = Normalizer(std_multiplier=std_multiplier)

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Identity — OlmoEarth's internal Normalizer handles raw values."""
        return images

    def _pad_group(
        self,
        g_images: torch.Tensor,
        dst_indices: list[int],
        target_channels: int,
    ) -> torch.Tensor:
        """Place each input channel at its OlmoEarth modality position.

        Returns ``(B, target_channels, H, W)`` with zeros for missing bands.
        """
        B, _, H, W = g_images.shape
        out = torch.zeros(B, target_channels, H, W, device=g_images.device, dtype=g_images.dtype)
        for local_idx, dst_idx in enumerate(dst_indices):
            out[:, dst_idx] = g_images[:, local_idx]
        return out

    def _normalize_with_band_stats(
        self,
        g_images: torch.Tensor,
        means: list[float],
        stds: list[float],
    ) -> torch.Tensor:
        """Per-band ``±std_multiplier·σ`` no-clip normalization from dataset stats.

        Maps each band's ``[mean - m·std, mean + m·std]`` to ``[0, 1]`` (no
        clipping), matching OlmoEarth's pretraining scheme / helios'
        ``NORM_NO_CLIP_2_STD`` but using the input's own ``BandSpec`` stats.
        ``g_images`` is ``(B, Csrc, H, W)`` in source-band order.
        """
        m = self.std_multiplier
        mean_t = torch.tensor(means, dtype=g_images.dtype, device=g_images.device).view(1, -1, 1, 1)
        std_t = torch.tensor(stds, dtype=g_images.dtype, device=g_images.device).view(1, -1, 1, 1)
        low = mean_t - m * std_t
        span = (2.0 * m * std_t).clamp(min=1e-6)
        return (g_images - low) / span

    def _to_nhwtc(self, g_images: torch.Tensor) -> np.ndarray:
        """``(B, C, H, W)`` tensor -> ``(B, H, W, T, C)`` numpy, replicating frames.

        ``T`` is ``self.time_steps``; for ``time_steps > 1`` the single input
        frame is repeated into each temporal slot.
        """
        g_nhwc = g_images.permute(0, 2, 3, 1).cpu().numpy()
        g_nhwtc = g_nhwc[:, :, :, None, :]
        if self.time_steps > 1:
            g_nhwtc = np.repeat(g_nhwtc, self.time_steps, axis=3)
        return g_nhwtc

    def _build_mask(
        self,
        B: int,
        H: int,
        W: int,
        num_band_sets: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-band-set ``(B, H, W, T, num_band_sets)`` mask — all visible.

        ``MaskValue.ONLINE_ENCODER`` is 0 so zeros == visible.  Shape must
        be 5-D because the encoder's ``apply_embedding_to_modality`` does
        ``mask[..., idx]`` for each band-set index.
        """
        return torch.zeros(
            B, H, W, self.time_steps, num_band_sets, dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Extract image-level embeddings from raw inputs."""
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        device = images.device
        B, _, H, W = images.shape

        if self.min_image_size is not None and (self.min_image_size > H or self.min_image_size > W):
            new_h = max(H, self.min_image_size)
            new_w = max(W, self.min_image_size)
            images = F.interpolate(
                images, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            B, _, H, W = images.shape

        # v1.1 uses linear patch embed which requires H,W divisible by patch_size.
        # Pad to the next multiple if needed (zero-padding is mask-safe).
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h))
            B, _, H, W = images.shape

        timestamps = torch.zeros(B, self.time_steps, 3, dtype=torch.long, device=device)
        timestamps[:, :, 0] = 15
        timestamps[:, :, 1] = 6
        timestamps[:, :, 2] = 2020

        sample_kwargs: dict = {}
        for group in self._sensor_groups:
            # Extract this sensor's channels from the full input tensor.
            g_images = images[:, group["src_indices"]]  # (B, Csensor, H, W)

            if self.norm_from_pretrained:
                # Rescale to S2 DN unless the sensor is a passthrough type, then
                # apply OlmoEarth's pretrained per-modality Normalizer.
                input_unit = group["input_unit"]
                if input_unit is not None:
                    g_images = to_s2_dn(g_images, input_unit)
                    if group["sensor"] == "landsat" and self.landsat_scale_factor is not None:
                        g_images = g_images * self.landsat_scale_factor
                if group["sensor"] == "sar" and self.sar_log_scale:
                    g_images = 10.0 * torch.log10(g_images.clamp(min=1e-6))

                g_images = self._pad_group(g_images, group["dst_indices"], group["channels"])
                g_nhwtc = self._to_nhwtc(g_images)
                g_nhwtc = self.normalizer.normalize(group["modality"], g_nhwtc)
            else:
                # Dataset-specific normalization: map each band's
                # [mean - m·std, mean + m·std] to [0, 1] (no clip) using its own
                # BandSpec stats, the same scheme OlmoEarth saw in pretraining.
                # Required when the input scale doesn't match the pretrained
                # normalizer (e.g. GeoBench uint8 Landsat).  No DN rescale.
                g_images = self._normalize_with_band_stats(
                    g_images, group["src_means"], group["src_stds"]
                )
                g_images = self._pad_group(g_images, group["dst_indices"], group["channels"])
                g_nhwtc = self._to_nhwtc(g_images)

            # Impute missing channels *after* normalization so each imputed
            # channel carries the normalized value of its source band. Doing
            # it post-norm avoids applying the wrong per-band statistics to a
            # borrowed band (the issue helios works around by re-imputing
            # after normalization); the result is that the imputed channel is
            # statistically identical to its source in normalized space.
            for src_dst, tgt_dst in group["impute_ops"]:
                g_nhwtc[..., tgt_dst] = g_nhwtc[..., src_dst]

            field = group["sample_field"]
            mask = self._build_mask(B, H, W, group["num_band_sets"], device)
            sample_kwargs[field] = torch.from_numpy(g_nhwtc).float().to(device)
            sample_kwargs[f"{field}_mask"] = mask

        sample = MaskedOlmoEarthSample(timestamps=timestamps, **sample_kwargs)

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
