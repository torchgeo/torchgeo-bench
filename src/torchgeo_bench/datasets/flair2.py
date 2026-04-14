"""FLAIR2 (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class FLAIR2(BenchDataset):
    """Aerial land-cover segmentation (13 classes).

    French aerial imagery with RGB, NIR, and elevation bands.
    """

    name = "flair2"
    task = "segmentation"
    num_classes = 13
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 4049, "val": 1022, "test": 3022}

    bands = [
        BandSpec("aerial", "red", "red", mean=110.305, std=50.71),
        BandSpec("aerial", "green", "green", mean=114.7908, std=44.3165),
        BandSpec("aerial", "blue", "blue", mean=105.6127, std=43.2948),
        BandSpec("aerial", "nir", "nir", mean=104.3409, std=39.0496),
        BandSpec("aerial", "elevation", "elevation", mean=17.6965, std=29.9427),
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

        return gb_v2.GeoBenchFLAIR2(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
