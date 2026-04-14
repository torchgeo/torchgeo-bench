"""MBrickKiln (GeoBench V1) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class MBrickKiln(BenchDataset):
    """Sentinel-2 brick kiln detection (2 classes).

    Based on the Brick-Kiln dataset with 13 Sentinel-2 spectral bands.
    """

    name = "m-brick-kiln"
    task = "classification"
    num_classes = 2
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 15063, "val": 999, "test": 999}

    bands = [
        BandSpec("s2", "coastal_aerosol", "01 - Coastal aerosol", wavelength_um=0.443, mean=574.76, std=193.61),
        BandSpec("s2", "blue", "02 - Blue", wavelength_um=0.49, mean=674.35, std=238.75),
        BandSpec("s2", "green", "03 - Green", wavelength_um=0.56, mean=886.37, std=276.96),
        BandSpec("s2", "red", "04 - Red", wavelength_um=0.665, mean=815.09, std=361.15),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", wavelength_um=0.705, mean=1128.81, std=364.59),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", wavelength_um=0.74, mean=1934.45, std=724.27),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", wavelength_um=0.783, mean=2045.77, std=819.65),
        BandSpec("s2", "nir", "08 - NIR", wavelength_um=0.842, mean=2012.74, std=794.37),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", wavelength_um=0.865, mean=1608.63, std=800.85),
        BandSpec("s2", "water_vapour", "09 - Water vapour", wavelength_um=0.945, mean=1129.82, std=704.02),
        BandSpec("s2", "swir_cirrus", "10 - SWIR - Cirrus", wavelength_um=1.375, mean=83.27, std=36.36),
        BandSpec("s2", "swir_1", "11 - SWIR", wavelength_um=1.61, mean=90.55, std=28.0),
        BandSpec("s2", "swir_2", "12 - SWIR", wavelength_um=2.19, mean=68.99, std=24.27),
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
