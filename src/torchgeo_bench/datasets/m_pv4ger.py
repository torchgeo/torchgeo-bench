"""MPv4ger (GeoBench V1) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class MPv4ger(BenchDataset):
    """Aerial solar panel detection (2 classes).

    Based on the PV4GER dataset with 3 aerial RGB bands.
    """

    name = "m-pv4ger"
    task = "classification"
    num_classes = 2
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 11814, "val": 999, "test": 999}

    bands = [
        BandSpec("aerial", "blue", "Blue", mean=116.63, std=44.67),
        BandSpec("aerial", "green", "Green", mean=119.66, std=48.28),
        BandSpec("aerial", "red", "Red", mean=113.39, std=54.2),
    ]

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = os.getenv("GEOBENCH_ROOT", "data/classification_v1.0")
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
        from ..geobench_dataset import GeoBenchDataset

        norm_arg: bool | str
        if normalize == "mean_stdev":
            norm_arg = True
        elif normalize == "none":
            norm_arg = False
        else:
            norm_arg = normalize

        v1_split = "valid" if split == "val" else split
        return GeoBenchDataset(
            root=self.root,
            dataset_name=self.name,
            split=v1_split,
            partition=partition,
            bands=bands,
            normalize=norm_arg,
            transform=transform,
        )
