"""Kuro Siwo (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class KuroSiwo(BenchDataset):
    """SAR flood mapping segmentation (4 classes).

    Uses VV, VH polarizations and a DEM band.
    """

    name = "kuro_siwo"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["vv", "vh", "dem"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    bands = [
        BandSpec("sar", "vv", "vv", mean=0.0953, std=0.0427),
        BandSpec("sar", "vh", "vh", mean=0.0264, std=0.0215),
        BandSpec("sar", "dem", "dem", mean=93.4313, std=1410.8382),
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

        return gb_v2.GeoBenchKuroSiwo(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
