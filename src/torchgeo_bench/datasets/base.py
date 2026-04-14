"""Base classes for torchgeo-bench dataset definitions.

Every benchmark dataset is a subclass of :class:`BenchDataset` that declares
its metadata (bands, number of classes, task type, split sizes) and knows
how to produce a PyTorch :class:`~torch.utils.data.Dataset` for each split.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BandSpec:
    """Metadata for a single spectral band in a dataset.

    Args:
        sensor: Sensor family identifier (e.g. ``"s2"``, ``"landsat"``,
            ``"aerial"``, ``"sar"``, ``"planet"``, ``"worldview"``).
        name: Canonical short band name used in the public API
            (e.g. ``"red"``, ``"b02"``, ``"nir"``, ``"vv"``).
        source_name: Band key as it appears in the data files.  For V1 HDF5
            files this is the long form (``"04 - Red"``); for V2 datasets
            this is typically the uppercase band code (``"B04"``).
        wavelength_um: Approximate centre wavelength in micrometres.
            ``None`` for non-optical bands (SAR, DEM, elevation).
        mean: Dataset-level mean pixel value.
        std: Dataset-level standard deviation.
    """

    sensor: str
    name: str
    source_name: str
    wavelength_um: float | None = None
    mean: float = 0.0
    std: float = 1.0


class BenchDataset(ABC):
    """Abstract base class for benchmark datasets.

    Subclasses must define the class-level metadata attributes listed below
    and implement :meth:`get_dataset`.

    Attributes:
        name: Dataset identifier used on the command line (e.g. ``"m-eurosat"``).
        task: ``"classification"`` or ``"segmentation"``.
        num_classes: Number of output classes.
        bands: Ordered list of all available spectral bands with statistics.
        rgb_bands: Short names of the bands to use for RGB-only mode.
        split_sizes: Number of samples per split for the *default* partition,
            keyed by ``"train"``, ``"val"``, ``"test"``.
        multilabel: Whether labels are multi-hot (e.g. BigEarthNet).
    """

    name: str
    task: Literal["classification", "segmentation"]
    num_classes: int
    bands: list[BandSpec]
    rgb_bands: list[str]
    split_sizes: dict[str, int]
    multilabel: bool = False

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_channels(self) -> int:
        """Total number of spectral bands."""
        return len(self.bands)

    @property
    def rgb_indices(self) -> list[int]:
        """Indices into :attr:`bands` for the RGB subset."""
        names = [b.name for b in self.bands]
        return [names.index(s) for s in self.rgb_bands if s in names]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @abstractmethod
    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        normalize: str = "mean_stdev",
    ) -> Dataset:
        """Return a PyTorch :class:`~torch.utils.data.Dataset` for a split.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            partition: Partition name (V1 only, e.g. ``"0.01x_train"``).
                Ignored by V2 datasets.
            bands: Tuple of canonical band names to load.  ``None`` loads all.
            transform: Optional sample transform callable.
            normalize: Normalization strategy — ``"mean_stdev"``,
                ``"min_max"``, ``"percentile_2_98"``, or ``"none"``.
        """
        raise NotImplementedError

    def get_dataloader(
        self,
        split: str,
        *,
        batch_size: int = 32,
        num_workers: int = 8,
        shuffle: bool | None = None,
        pin_memory: bool = True,
        **dataset_kwargs,
    ) -> DataLoader:
        """Convenience wrapper: build a :class:`~torch.utils.data.DataLoader`.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            batch_size: Batch size.
            num_workers: Number of dataloader worker processes.
            shuffle: Shuffle the data.  Defaults to ``True`` for train,
                ``False`` otherwise.
            pin_memory: Pin memory for CUDA transfers.
            **dataset_kwargs: Forwarded to :meth:`get_dataset`.
        """
        ds = self.get_dataset(split, **dataset_kwargs)
        if shuffle is None:
            shuffle = split == "train"
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
