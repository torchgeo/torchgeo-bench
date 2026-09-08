"""NWPU-RESISC45 aerial scene classification via torchgeo."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import RESISC45 as TGRESISC45
from torchvision.transforms import Compose

from .base import BandSpec, BenchDataset


class RESISC45(BenchDataset):
    """Aerial scene classification, 45 classes, via torchgeo.

    31,500 RGB images (700 per class) at 256x256, extracted from Google Earth by Northwestern Polytechnical University. Splits are torchgeo's published 60/20/20 partition (18,900 / 6,300 / 6,300).

    These 8-bit composites have no radiometric calibration, per-image geolocation, or scale metadata. Wavelengths are nominal visible-light centres, not measured sensor responses.

    Images span 0.2--30 m/px. The ``aerial`` tag assumes 1 m GSD so resolution-aware models (OlmoEarth, UniverSat) accept them as RGB aerial imagery; it is an approximation, not a fixed-resolution claim.
    """

    name = "resisc45"
    task = "classification"
    num_classes = 45
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 18900, "val": 6300, "test": 6300}
    supports_partitions = False

    # Statistics use the 18,900-image train split in raw 0-255 units (scripts/compute_band_statistics.py).
    # ``source_name`` identifies the RGB channel, not a band key in the JPEG.
    # fmt: off
    bands = [
        BandSpec("aerial", "red", "R", mean=93.8939, std=51.8492, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=97.1123, std=47.2366, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=87.5678, std=47.0631, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/resisc45")``; torchgeo manages the layout below."""
        return Path("data/resisc45")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the wrapped torchgeo dataset for the split.

        The upstream loader always reads all three RGB channels. Select bands before the caller's ``transform`` to avoid resizing unused channels.
        """
        del partition
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        select = _make_band_select(indices, len(self.bands))
        if select is not None:
            transform = select if transform is None else Compose([select, transform])
        return TGRESISC45(
            root=str(self.data_root()),
            split=split,
            transforms=transform,
        )


def _make_band_select(indices: list[int], n_bands: int) -> Callable[[dict], dict] | None:
    """Return a transform selecting ``indices`` from the channel axis.

    ``None`` when the selection is the identity (every band, in order), so the
    common ``bands="all"`` and ``bands="rgb"`` paths add no per-sample work.
    """
    if indices == list(range(n_bands)):
        return None
    index = torch.tensor(indices)

    def _select(sample: dict) -> dict:
        sample["image"] = sample["image"].index_select(-3, index)
        return sample

    return _select
