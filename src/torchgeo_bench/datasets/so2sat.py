"""So2Sat (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class So2Sat(BenchDataset):
    """Sentinel-2 + SAR local climate zone classification (17 classes).

    GeoBench V2 version with 10 Sentinel-2 and 2 SAR bands.
    """

    name = "so2sat"
    task = "classification"
    num_classes = 17
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 19992, "val": 986, "test": 986}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=0.1295, std=0.0414),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=0.1172, std=0.052),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=0.1138, std=0.0733),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=0.1272, std=0.0694),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=0.1707, std=0.0751),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=0.1928, std=0.0856),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=0.1855, std=0.0865),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=0.2073, std=0.094),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=0.1768, std=0.1024),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=0.1285, std=0.0923),
        BandSpec("sar", "vv", "VV", mean=-0.0, std=0.5443),
        BandSpec("sar", "vh", "VH", mean=-0.0, std=0.2156),
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

        return gb_v2.GeoBenchSo2Sat(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
