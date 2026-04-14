"""CaFFe (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class CaFFe(BenchDataset):
    """Aerial grayscale classification (4 classes).

    CaFFe dataset with a single grayscale aerial band.
    """

    name = "caffe"
    task = "classification"
    num_classes = 4
    multilabel = False
    rgb_bands = ["gray"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    bands = [
        BandSpec("aerial", "gray", "gray", mean=62.6825, std=79.8002),
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

        return gb_v2.GeoBenchCaFFe(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
