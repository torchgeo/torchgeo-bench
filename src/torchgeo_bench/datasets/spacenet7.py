"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class SpaceNet7(BenchDataset):
    """Planet building change segmentation (3 classes).

    RGB imagery from Planet satellites.
    """

    name = "spacenet7"
    task = "segmentation"
    num_classes = 3
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 3500, "val": 652, "test": 1152}

    bands = [
        BandSpec("planet", "red", "red", mean=116.9447, std=61.6558),
        BandSpec("planet", "green", "green", mean=103.5589, std=49.649),
        BandSpec("planet", "blue", "blue", mean=76.7743, std=45.8807),
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

        return gb_v2.GeoBenchSpaceNet7(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
