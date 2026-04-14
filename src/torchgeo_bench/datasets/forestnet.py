"""Forestnet (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class Forestnet(BenchDataset):
    """Sentinel-2 forest-change classification (12 classes).

    GeoBench V2 version with 6 Sentinel-2 spectral bands.
    """

    name = "forestnet"
    task = "classification"
    num_classes = 12
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 6464, "val": 989, "test": 993}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=72.3759, std=16.2839),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=83.1816, std=15.3587),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=77.0861, std=16.6665),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=123.5425, std=16.9485),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=91.0483, std=14.2801),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=74.3097, std=13.2854),
    ]

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = os.getenv("GEOBENCH_V2_ROOT", "data/geobenchv2")
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
        del partition, normalize
        import geobench_v2.datasets as gb_v2

        return gb_v2.GeoBenchForestnet(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
