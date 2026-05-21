"""RESISC45 (torchgeo) benchmark dataset wrapper."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import RESISC45 as TGRESISC45

from .base import BandSpec, BenchDataset


class _Resisc45DatasetView(Dataset):
    """Adapt torchgeo RESISC45 samples to selected RGB channels."""

    def __init__(
        self,
        base_dataset: Dataset,
        channel_indices: list[int],
        transform: Callable | None,
    ) -> None:
        self.base_dataset = base_dataset
        self.channel_indices = channel_indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.base_dataset[index]
        image = sample["image"][self.channel_indices].to(dtype=torch.float32)
        out = {"image": image, "label": sample["label"]}
        if self.transform is not None:
            out = self.transform(out)
        return out


class RESISC45(BenchDataset):
    """NWPU-RESISC45 scene classification dataset via torchgeo."""

    name = "resisc45"
    task = "classification"
    num_classes = 45
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 18900, "val": 6300, "test": 6300}
    supports_partitions = False

    # Train-set RGB statistics from torchgeo RESISC45DataModule.
    bands = [
        BandSpec("aerial", "red", "red", mean=93.89391792, std=51.84919672, min=0, max=255, wavelength_um=0.66),
        BandSpec("aerial", "green", "green", mean=97.11226906, std=47.2365918, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "blue", mean=87.56775284, std=47.06308786, min=0, max=255, wavelength_um=0.48),
    ]

    @classmethod
    def data_root(cls) -> Path:
        return Path("data/resisc45")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        del partition
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split!r}. Expected 'train', 'val', or 'test'.")

        base_dataset = TGRESISC45(
            root=str(self.data_root()),
            split=split,
            transforms=None,
            download=True,
        )

        channel_lookup = {spec.name: idx for idx, spec in enumerate(self.bands)}
        selected_specs = self.select_band_specs(bands)
        selected_channels = [channel_lookup[spec.name] for spec in selected_specs]
        return _Resisc45DatasetView(base_dataset, selected_channels, transform)
