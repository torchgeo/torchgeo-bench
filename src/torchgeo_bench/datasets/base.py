"""Base classes for torchgeo-bench dataset definitions.

Every benchmark dataset is a subclass of :class:`BenchDataset` that declares
its metadata (bands, number of classes, task type, split sizes) and knows how
to produce a PyTorch :class:`~torch.utils.data.Dataset` for each split.

Datasets always live under ``data/`` (relative to the current working
directory). Each family base class (``_V1Dataset``, ``_V2Dataset``) and the
torchgeo :class:`~torchgeo_bench.datasets.eurosat.EuroSAT` wrapper exposes its
own :meth:`BenchDataset.data_root` returning the family-specific subdirectory.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
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
        source_name: Band key as it appears in the data files. For V1 HDF5
            files this is the long form (``"04 - Red"``); for V2 datasets
            this is typically the uppercase band code (``"B04"``).
        mean: Train-split mean pixel value (raw units, no normalization).
        std: Train-split standard deviation.
        min: Train-split minimum pixel value.
        max: Train-split maximum pixel value.
        wavelength_um: Approximate centre wavelength in micrometres.
            ``None`` for non-optical bands (SAR, DEM, elevation).
    """

    sensor: str
    name: str
    source_name: str
    mean: float
    std: float
    min: float
    max: float
    wavelength_um: float | None = None


class BenchDataset(ABC):
    """Abstract base class for benchmark datasets.

    Subclasses must define the class-level metadata attributes listed below
    and implement :meth:`get_dataset` and :meth:`data_root`.

    Attributes:
        name: Dataset identifier used on the command line (e.g. ``"m-eurosat"``).
        task: ``"classification"`` or ``"segmentation"``.
        num_classes: Number of output classes.
        bands: Ordered list of all available spectral bands with statistics.
        rgb_bands: Short names of the bands to use for RGB-only mode.
        split_sizes: Number of samples per split for the *default* partition,
            keyed by ``"train"``, ``"val"``, ``"test"``.
        multilabel: Whether labels are multi-hot (e.g. BigEarthNet).
        supports_partitions: Whether the dataset honours a non-default
            ``partition`` argument (V1 GeoBench datasets do; V2 does not).
    """

    name: str
    task: Literal["classification", "segmentation"]
    num_classes: int
    bands: list[BandSpec]
    rgb_bands: list[str]
    split_sizes: dict[str, int]
    multilabel: bool = False
    supports_partitions: bool = False

    @property
    def num_channels(self) -> int:
        """Total number of spectral bands."""
        return len(self.bands)

    @property
    def rgb_indices(self) -> list[int]:
        """Indices into :attr:`bands` for the RGB subset."""
        names = [b.name for b in self.bands]
        return [names.index(s) for s in self.rgb_bands if s in names]

    @classmethod
    @abstractmethod
    def data_root(cls) -> Path:
        """Return the directory the upstream loader expects.

        For V1/V2 wrappers this is the *parent* directory containing per-dataset
        subdirectories (e.g. ``data/classification_v1.0``); for torchgeo
        wrappers it is the dataset's own root (e.g. ``data/eurosat``).
        """

    def select_band_specs(self, bands: Iterable[str] | None) -> list[BandSpec]:
        """Return the :class:`BandSpec` entries matching *bands*.

        Preserves the order given by *bands*. Raises ``ValueError`` if any
        requested band is not declared on the dataset.
        """
        if bands is None:
            return list(self.bands)
        by_name = {b.name: b for b in self.bands}
        result: list[BandSpec] = []
        for name in bands:
            if name not in by_name:
                raise ValueError(
                    f"{type(self).__name__}: unknown band {name!r}; available: {sorted(by_name)}"
                )
            result.append(by_name[name])
        return result

    @abstractmethod
    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a PyTorch :class:`~torch.utils.data.Dataset` for a split.

        Datasets always emit raw float32 values; normalization is the
        :class:`~torchgeo_bench.models.interface.BenchModel`'s job.

        Args:
            split: ``"train"``, ``"val"``, or ``"test"``.
            partition: Partition name (V1 only, e.g. ``"0.01x_train"``).
                Ignored by datasets where :attr:`supports_partitions` is
                ``False``.
            bands: Tuple of canonical band names to load. ``None`` loads all.
            transform: Optional sample transform callable.
        """

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
        """Convenience wrapper: build a :class:`~torch.utils.data.DataLoader`."""
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
