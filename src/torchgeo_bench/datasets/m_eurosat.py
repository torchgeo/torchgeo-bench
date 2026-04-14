"""MEurosat (GeoBench V1) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class MEurosat(BenchDataset):
    """Sentinel-2 land-use classification (10 classes).

    Based on the EuroSAT dataset with 13 Sentinel-2 spectral bands.
    """

    name = "m-eurosat"
    task = "classification"
    num_classes = 10
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 2000, "val": 1000, "test": 1000}

    bands = [
        BandSpec("s2", "coastal_aerosol", "01 - Coastal aerosol", wavelength_um=0.443, mean=1356.78, std=246.34),
        BandSpec("s2", "blue", "02 - Blue", wavelength_um=0.49, mean=1123.14, std=334.28),
        BandSpec("s2", "green", "03 - Green", wavelength_um=0.56, mean=1057.28, std=392.23),
        BandSpec("s2", "red", "04 - Red", wavelength_um=0.665, mean=959.18, std=590.94),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", wavelength_um=0.705, mean=1227.44, std=548.57),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", wavelength_um=0.74, mean=2076.63, std=843.08),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", wavelength_um=0.783, mean=2463.43, std=1071.73),
        BandSpec("s2", "nir", "08 - NIR", wavelength_um=0.842, mean=2390.32, std=1106.87),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", wavelength_um=0.865, mean=761.55, std=404.41),
        BandSpec("s2", "water_vapour", "09 - Water vapour", wavelength_um=0.945, mean=12.33, std=5.08),
        BandSpec("s2", "swir_cirrus", "10 - SWIR - Cirrus", wavelength_um=1.375, mean=1861.29, std=963.03),
        BandSpec("s2", "swir_1", "11 - SWIR", wavelength_um=1.61, mean=1138.94, std=742.79),
        BandSpec("s2", "swir_2", "12 - SWIR", wavelength_um=2.19, mean=2699.78, std=1215.04),
    ]

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = os.getenv("GEOBENCH_ROOT", "data/classification_v1.0")
        super().__init__(root)

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        normalize: str = "mean_stdev",
    ) -> Dataset:
        """Return a PyTorch Dataset for the given split."""
        from ..geobench_dataset import GeoBenchDataset

        norm_arg: bool | str
        if normalize == "mean_stdev":
            norm_arg = True
        elif normalize == "none":
            norm_arg = False
        else:
            norm_arg = normalize

        v1_split = "valid" if split == "val" else split
        return GeoBenchDataset(
            root=self.root,
            dataset_name=self.name,
            split=v1_split,
            partition=partition,
            bands=bands,
            normalize=norm_arg,
            transform=transform,
        )
