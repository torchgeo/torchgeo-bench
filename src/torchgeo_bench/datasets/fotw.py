"""Fields of the World (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class FieldsOfTheWorld(BenchDataset):
    """Sentinel-2 field boundary segmentation (4 classes).

    Classes: background, field, boundary, other.
    """

    name = "fotw"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    bands = [
        BandSpec("s2", "red", "red", mean=862.084, std=681.1667),
        BandSpec("s2", "green", "green", mean=853.3895, std=508.6401),
        BandSpec("s2", "blue", "blue", mean=592.008, std=454.0239),
        BandSpec("s2", "nir", "nir", mean=2984.3018, std=1043.6527),
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

        return gb_v2.GeoBenchFieldsOfTheWorld(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
