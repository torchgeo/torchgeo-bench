"""SpaceNet2 (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class SpaceNet2(BenchDataset):
    """WorldView building footprint segmentation (3 classes).

    8 multispectral + 1 panchromatic band from WorldView satellite.
    """

    name = "spacenet2"
    task = "segmentation"
    num_classes = 3
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 5186, "val": 1461, "test": 2961}

    bands = [
        BandSpec("worldview", "coastal", "coastal", mean=298.7281, std=106.9792),
        BandSpec("worldview", "blue", "blue", mean=358.0099, std=148.1868),
        BandSpec("worldview", "green", "green", mean=464.5104, std=224.4095),
        BandSpec("worldview", "yellow", "yellow", mean=419.9473, std=225.7901),
        BandSpec("worldview", "red", "red", mean=333.6004, std=194.0233),
        BandSpec("worldview", "red_edge", "red_edge", mean=408.6689, std=208.4557),
        BandSpec("worldview", "nir1", "nir1", mean=475.0842, std=234.7585),
        BandSpec("worldview", "nir2", "nir2", mean=362.3487, std=193.2321),
        BandSpec("worldview", "pan", "pan", mean=468.574, std=260.8954),
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

        return gb_v2.GeoBenchSpaceNet2(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
