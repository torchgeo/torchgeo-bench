"""MBigEarthNet (GeoBench V1) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class MBigEarthNet(BenchDataset):
    """Sentinel-2 multi-label land-cover classification (43 classes).

    Based on the BigEarthNet dataset with 12 Sentinel-2 spectral bands.
    Uses multi-hot label encoding.
    """

    name = "m-bigearthnet"
    task = "classification"
    num_classes = 43
    multilabel = True
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 20000, "val": 1000, "test": 1000}

    bands = [
        BandSpec("s2", "coastal_aerosol", "01 - Coastal aerosol", wavelength_um=0.443, mean=386.65, std=467.31),
        BandSpec("s2", "blue", "02 - Blue", wavelength_um=0.49, mean=488.99, std=510.79),
        BandSpec("s2", "green", "03 - Green", wavelength_um=0.56, mean=714.61, std=551.8),
        BandSpec("s2", "red", "04 - Red", wavelength_um=0.665, mean=738.26, std=691.78),
        BandSpec("s2", "red_edge_1", "05 - Vegetation Red Edge", wavelength_um=0.705, mean=1114.42, std=700.45),
        BandSpec("s2", "red_edge_2", "06 - Vegetation Red Edge", wavelength_um=0.74, mean=1910.19, std=976.75),
        BandSpec("s2", "red_edge_3", "07 - Vegetation Red Edge", wavelength_um=0.783, mean=2191.48, std=1134.89),
        BandSpec("s2", "nir", "08 - NIR", wavelength_um=0.842, mean=2334.29, std=1238.07),
        BandSpec("s2", "red_edge_4", "08A - Vegetation Red Edge", wavelength_um=0.865, mean=2392.91, std=1215.98),
        BandSpec("s2", "water_vapour", "09 - Water vapour", wavelength_um=0.945, mean=2367.29, std=1153.86),
        BandSpec("s2", "swir_1", "11 - SWIR", wavelength_um=1.61, mean=1902.69, std=1117.01),
        BandSpec("s2", "swir_2", "12 - SWIR", wavelength_um=2.19, mean=1261.07, std=894.74),
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
