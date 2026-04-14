"""PASTIS (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class PASTIS(BenchDataset):
    """Sentinel-2 + SAR crop type segmentation (20 classes).

    Includes ascending and descending SAR orbit passes.
    """

    name = "pastis"
    task = "segmentation"
    num_classes = 20
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 1455, "val": 482, "test": 496}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=1369.9984, std=2247.7554),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=1583.1479, std=2179.1699),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=1627.6497, std=2255.1763),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=1930.8378, std=2142.7222),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=2921.8389, std=1928.733),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=3284.9307, std=1900.8661),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=3421.7988, std=1890.3164),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=3544.2336, std=1873.0812),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=2564.7144, std=1409.2015),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=1708.5986, std=1189.0947),
        BandSpec("sar", "vv_asc", "VV_asc", mean=-10.2839, std=3.0927),
        BandSpec("sar", "vh_asc", "VH_asc", mean=-16.8657, std=3.0265),
        BandSpec("sar", "vv_vh_asc", "VV/VH_asc", mean=6.5818, std=3.3432),
        BandSpec("sar", "vv_desc", "VV_desc", mean=-10.3489, std=3.2165),
        BandSpec("sar", "vh_desc", "VH_desc", mean=-16.9022, std=3.0307),
        BandSpec("sar", "vv_vh_desc", "VV/VH_desc", mean=6.5533, std=3.3312),
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

        return gb_v2.GeoBenchPASTIS(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
