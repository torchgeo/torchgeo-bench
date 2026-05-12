"""Detect a dataset's input unit and convert between scales for model-faithful normalisation.

Each pretrained backbone was trained on inputs in a specific scale (raw S2
DN, reflectance in [0, 1], uint8/255, etc.).  Wrappers declare their
expected unit and use these helpers to bring whatever the dataset emits
into that scale before applying any model-specific per-band normalisation.
"""

from enum import StrEnum

import torch

from torchgeo_bench.datasets.base import BandSpec


class InputUnit(StrEnum):
    """Coarse buckets for image-tensor scales we encounter in GeoBench."""

    S2_DN = "s2_dn"  # raw Sentinel-2 sensor counts, 0..~10000+
    REFLECTANCE_0_1 = "reflectance_0_1"  # already-normalised, ~0..1 (2.8 max in m-so2sat)
    UINT8 = "uint8"  # 0..255 NAIP / Landsat L1


def detect_input_unit(bands: list[BandSpec]) -> InputUnit:
    """Guess the source unit from per-band ``max`` magnitudes.

    Heuristic:

    * any optical band with ``max > 1000`` -> :attr:`S2_DN`
    * else any optical band with ``max > 10`` -> :attr:`UINT8`
    * otherwise -> :attr:`REFLECTANCE_0_1`
    """
    optical = [b for b in bands if b.wavelength_um is not None] or bands
    max_max = max(b.max for b in optical)
    if max_max > 1000:
        return InputUnit.S2_DN
    if max_max > 10:
        return InputUnit.UINT8
    return InputUnit.REFLECTANCE_0_1


def to_reflectance(images: torch.Tensor, src: InputUnit) -> torch.Tensor:
    """Bring values into roughly ``[0, 1]`` reflectance space."""
    if src == InputUnit.S2_DN:
        return images / 10000.0
    if src == InputUnit.UINT8:
        return images / 255.0
    return images


def to_s2_dn(images: torch.Tensor, src: InputUnit) -> torch.Tensor:
    """Bring values into S2 DN scale (~``[0, 10000]``)."""
    if src == InputUnit.S2_DN:
        return images
    if src == InputUnit.REFLECTANCE_0_1:
        return images * 10000.0
    # UINT8 to DN: rescale [0, 255] -> [0, 10000].
    return images * (10000.0 / 255.0)
