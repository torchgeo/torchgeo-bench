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


def map_to_model_bands(
    images: torch.Tensor,
    src_bands: list[BandSpec],
    target_band_names: list[str],
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
    src_index: dict[str, int] = {}
    for i, b in enumerate(src_bands):
        src_index.setdefault(canonical_band_name(b.name), i)

    B, _, H, W = images.shape
    out = torch.zeros(B, len(target_band_names), H, W, device=images.device, dtype=images.dtype)
    missing: list[bool] = []
    for j, name in enumerate(target_band_names):
        idx = src_index.get(canonical_band_name(name))
        if idx is None:
            missing.append(True)
            continue
        out[:, j] = images[:, idx]
        missing.append(False)
    return out, missing


def wavelengths_um(bands: list[BandSpec], default_um: float = 0.6) -> list[float]:
    """Return per-band centre wavelengths in micrometres, filling ``None`` with ``default_um``."""
    return [float(b.wavelength_um) if b.wavelength_um is not None else default_um for b in bands]
