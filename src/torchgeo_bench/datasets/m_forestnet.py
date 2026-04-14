"""MForestnet (GeoBench V1) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class MForestnet(BenchDataset):
    """Landsat forest-change classification (12 classes).

    Based on the ForestNet dataset with 6 Landsat spectral bands.
    """

    name = "m-forestnet"
    task = "classification"
    num_classes = 12
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 6464, "val": 989, "test": 993}

    bands = [
        BandSpec("landsat", "blue", "02 - Blue", wavelength_um=0.49, mean=72.85, std=15.84),
        BandSpec("landsat", "green", "03 - Green", wavelength_um=0.56, mean=83.68, std=14.79),
        BandSpec("landsat", "red", "04 - Red", wavelength_um=0.665, mean=77.58, std=16.1),
        BandSpec("landsat", "nir", "05 - NIR", mean=123.99, std=16.35),
        BandSpec("landsat", "swir_1", "06 - SWIR1", mean=91.54, std=13.79),
        BandSpec("landsat", "swir_2", "07 - SWIR2", mean=74.72, std=12.69),
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
