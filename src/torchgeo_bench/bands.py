"""Spectral band metadata shared by datasets and models.

:class:`BandSpec` lives in this dependency-free leaf module (stdlib only) so
that ``from torchgeo_bench.bands import BandSpec`` — and the re-export in
:mod:`torchgeo_bench.datasets.base` — never pays for torch/torchgeo imports.
"""

from dataclasses import dataclass

__all__ = ["BandSpec"]


@dataclass(frozen=True)
class BandSpec:
    """Metadata for a single spectral band in a dataset.

    Args:
        sensor: Sensor family identifier (e.g. ``"s2"``, ``"landsat"``,
            ``"aerial"``, ``"sar"``, ``"planet"``, ``"worldview"``).
        name: Canonical short band name used in the public API
            (e.g. ``"red"``, ``"b02"``, ``"nir"``, ``"vv"``).
        source_name: Band key as it appears in the data files. For V1 HDF5
            files this is the long form (``"04 - Red"``); for V2 datasets
            this is typically the uppercase band code (``"B04"``).
        mean: Train-split mean pixel value (raw units, no normalization).
        std: Train-split standard deviation.
        min: Train-split minimum pixel value.
        max: Train-split maximum pixel value.
        wavelength_um: Approximate centre wavelength in micrometres.
            ``None`` for non-optical bands (SAR, DEM, elevation).
    """

    sensor: str
    name: str
    source_name: str
    mean: float
    std: float
    min: float
    max: float
    wavelength_um: float | None = None
