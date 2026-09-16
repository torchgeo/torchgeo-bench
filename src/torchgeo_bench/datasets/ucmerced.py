# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""UC Merced Land Use classification via torchgeo."""

from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

from torch.utils.data import Dataset
from torchgeo.datasets import UCMerced as TGUCMerced
from torchgeo.datasets.utils import Sample
from torchvision.transforms import Compose

from .base import BandSpec, BenchDataset


class UCMerced(BenchDataset):
    """Aerial land use classification, 21 classes, via torchgeo.

    2,100 RGB images from USGS urban area imagery at 1 ft (0.3048 m) resolution.
    Uses torchgeo's published 60/20/20 split: 1,260 train, 420 validation, 420 test.
    Torchgeo resizes the few non-square images to 256x256 before caller transforms.

    Wavelengths are nominal visible-light centres, not measured sensor responses.
    The shared ``aerial`` sensor tag uses an approximate 1 m GSD for model routing.
    """

    name = "ucmerced"
    task = "classification"
    num_classes = 21
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["red", "green", "blue"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 1260, "val": 420, "test": 420}
    supports_partitions = False

    # Train-split statistics in raw 0-255 units from scripts/compute_band_statistics.py.
    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("aerial", "red", "R", mean=122.6146, std=55.8084, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=124.0959, std=51.7905, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=114.2279, std=50.0025, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/ucmerced")``; torchgeo manages the layout below."""
        return Path("data/ucmerced")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[Sample], Sample] | None = None,
    ) -> Dataset:
        """Return a torchgeo split with requested bands selected before transforms."""
        del partition
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        if indices != list(range(len(self.bands))):
            select = _SelectBands(indices)
            transform = select if transform is None else Compose([select, transform])
        return TGUCMerced(root=str(self.data_root()), split=split, transforms=transform)


class _SelectBands:
    """Select RGB channels with a transform picklable by spawned data workers."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def __call__(self, sample: Sample) -> Sample:
        sample["image"] = sample["image"][self.indices]
        return sample
