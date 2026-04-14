"""Burn Scars (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class BurnScars(BenchDataset):
    """Sentinel-2 burn scar segmentation (3 classes).

    Classes: background, burn, cloud.
    """

    name = "burn_scars"
    task = "segmentation"
    num_classes = 3
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 524, "val": 160, "test": 120}

    bands = [
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=0.0333, std=0.0227),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=0.057, std=0.0268),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=0.0589, std=0.04),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=0.2323, std=0.0779),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=0.1973, std=0.0871),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=0.1194, std=0.0724),
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

        return gb_v2.GeoBenchBurnScars(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
