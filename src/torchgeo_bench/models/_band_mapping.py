"""Map dataset BandSpec lists onto pretrained-model band slots."""

import logging

import torch

from torchgeo_bench.datasets.base import BandSpec

logger = logging.getLogger(__name__)


_RGB_ALIASES = {
    "red": "red",
    "r": "red",
    "b04": "red",
    "04": "red",
    "green": "green",
    "g": "green",
    "b03": "green",
    "03": "green",
    "blue": "blue",
    "b": "blue",
    "b02": "blue",
    "02": "blue",
}

_NIR_ALIASES = {
    "nir": "nir",
    "b08": "nir",
    "08": "nir",
    "nir_narrow": "nir_narrow",
    "red_edge_4": "nir_narrow",
    "rededge4": "nir_narrow",
    "b8a": "nir_narrow",
    "8a": "nir_narrow",
}

_SWIR_ALIASES = {
    "swir1": "swir1",
    "swir_1": "swir1",
    "b11": "swir1",
    "11": "swir1",
    "swir2": "swir2",
    "swir_2": "swir2",
    "b12": "swir2",
    "12": "swir2",
}

_S2_EXTRA = {
    "b01": "coastal",
    "01": "coastal",
    "coastal": "coastal",
    "coastal_aerosol": "coastal",
    "b05": "rededge1",
    "05": "rededge1",
    "rededge1": "rededge1",
    "red_edge_1": "rededge1",
    "b06": "rededge2",
    "06": "rededge2",
    "rededge2": "rededge2",
    "red_edge_2": "rededge2",
    "b07": "rededge3",
    "07": "rededge3",
    "rededge3": "rededge3",
    "red_edge_3": "rededge3",
    "b09": "watervapor",
    "09": "watervapor",
    "watervapor": "watervapor",
    "water_vapour": "watervapor",
    "water_vapor": "watervapor",
    "b10": "cirrus",
    "10": "cirrus",
    "cirrus": "cirrus",
    "swir_cirrus": "cirrus",
}

_SAR = {"vv": "vv", "vh": "vh", "hh": "hh", "hv": "hv"}

_BAND_ALIASES: dict[str, str] = {
    **_RGB_ALIASES,
    **_NIR_ALIASES,
    **_SWIR_ALIASES,
    **_S2_EXTRA,
    **_SAR,
}


def canonical_band_name(name: str) -> str:
    """Map an input band name to the canonical short name."""
    key = name.strip().lower().replace(" ", "")
    head = key.split("-")[0]
    if head in _BAND_ALIASES:
        return _BAND_ALIASES[head]
    return _BAND_ALIASES.get(key, key)


def resolve_src_indices(
    src_bands: list[BandSpec],
    *,
    preferred_sensors: tuple[str, ...] = (),
) -> dict[str, int]:
    """Map each canonical band name to one source channel index.

    When two source bands share a canonical name (e.g. TreeSatAI's aerial
    ``red`` vs Sentinel-2 ``b04``), the band whose ``BandSpec.sensor``
    appears earliest in ``preferred_sensors`` wins; ties keep the first
    occurrence in ``src_bands``.
    """
    best: dict[str, tuple[int, int]] = {}
    for i, b in enumerate(src_bands):
        name = canonical_band_name(b.name)
        rank = (
            preferred_sensors.index(b.sensor)
            if b.sensor in preferred_sensors
            else len(preferred_sensors)
        )
        if name not in best or rank < best[name][0]:
            best[name] = (rank, i)
    return {name: i for name, (_, i) in best.items()}


def select_src_bands(
    src_bands: list[BandSpec],
    target_band_names: list[str],
    *,
    preferred_sensors: tuple[str, ...] = (),
) -> tuple[list[int], list[str]]:
    """Select source channel indices for the targets present in ``src_bands``.

    Unlike :func:`map_to_model_bands` this never pads: targets with no
    matching source band are simply dropped, so callers can hand the
    surviving subset to a model-native band-selection API (e.g. TerraTorch's
    ``bands=`` argument for TerraMind).

    Returns ``(indices, selected)`` where ``indices[i]`` is the source
    channel for ``selected[i]`` and ``selected`` preserves the order of
    ``target_band_names``.
    """
    src_index = resolve_src_indices(src_bands, preferred_sensors=preferred_sensors)
    indices: list[int] = []
    selected: list[str] = []
    for name in target_band_names:
        idx = src_index.get(canonical_band_name(name))
        if idx is not None:
            indices.append(idx)
            selected.append(name)
    if not indices:
        available = sorted(src_index)
        raise ValueError(
            f"select_src_bands: none of the target bands {target_band_names} are present. "
            f"Available canonical bands: {available}."
        )
    return indices, selected


def map_to_model_bands(
    images: torch.Tensor,
    src_bands: list[BandSpec],
    target_band_names: list[str],
    *,
    allow_missing: bool = False,
    preferred_sensors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, list[bool]]:
    """Rearrange ``images`` from src band order to ``target_band_names``, zero-filling gaps.

    Returns ``(mapped, missing)`` where ``missing[i]`` is True iff slot
    ``i`` was zero-filled.
    """
    if images.shape[1] != len(src_bands):
        raise ValueError(
            f"map_to_model_bands: images has {images.shape[1]} channels but "
            f"src_bands has {len(src_bands)} entries."
        )
    src_index = resolve_src_indices(src_bands, preferred_sensors=preferred_sensors)

    B, _, H, W = images.shape
    out = torch.zeros(B, len(target_band_names), H, W, device=images.device, dtype=images.dtype)
    missing: list[bool] = []
    for j, name in enumerate(target_band_names):
        idx = src_index.get(canonical_band_name(name))
        if idx is None:
            if not allow_missing:
                available = [canonical_band_name(b.name) for b in src_bands]
                raise ValueError(
                    f"Missing required model band {name!r}. Available canonical bands: "
                    f"{available}. Pass allow_missing=True only for an explicit zero-fill ablation."
                )
            missing.append(True)
            continue
        out[:, j] = images[:, idx]
        missing.append(False)
    return out, missing


def wavelengths_um(bands: list[BandSpec], default_um: float | None = None) -> list[float]:
    """Return per-band centre wavelengths in micrometres.

    Missing wavelengths raise by default. Passing ``default_um`` is an explicit
    opt-in for callers running a known fallback ablation.
    """
    missing = [b.name for b in bands if b.wavelength_um is None]
    if missing and default_um is None:
        raise ValueError(
            f"Missing wavelengths for {missing}. Pass explicit wavelengths or default_um "
            "only for a deliberate fallback ablation."
        )
    return [
        float(b.wavelength_um) if b.wavelength_um is not None else float(default_um) for b in bands
    ]
