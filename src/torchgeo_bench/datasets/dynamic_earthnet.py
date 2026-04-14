"""Dynamic EarthNet (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class DynamicEarthNet(BenchDataset):
    """Planet + Sentinel-2 land-cover change segmentation (7 classes)."""

    name = "dynamic_earthnet"
    task = "segmentation"
    num_classes = 7
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 700, "val": 100, "test": 200}

    bands = [
        BandSpec("planet", "b", "b", mean=641.1243, std=523.4901),
        BandSpec("planet", "g", "g", mean=881.2557, std=647.627),
        BandSpec("planet", "r", "r", mean=1011.3513, std=888.1036),
        BandSpec("planet", "nir", "nir", mean=2609.9226, std=992.0602),
        BandSpec("s2", "b01", "B01", wavelength_um=0.443, mean=1091.7622, std=1414.6219),
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=1318.8528, std=1343.7621),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=1380.1472, std=1427.9449),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=2678.5251, std=1376.4869),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=1730.9559, std=1429.6456),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=2373.4131, std=1333.8411),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=2630.0532, std=1370.478),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=2782.6868, std=1386.9127),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=2307.1587, std=1394.8506),
        BandSpec("s2", "b10", "B10", wavelength_um=1.375, mean=1719.8887, std=1304.7115),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=1003.9291, std=1475.8456),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=3031.0217, std=2124.4131),
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

        return gb_v2.GeoBenchDynamicEarthNet(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
