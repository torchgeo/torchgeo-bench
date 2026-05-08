"""GeoBench V1 PyTorch :class:`Dataset` and per-wrapper base class.

Lightweight HDF5 reader that does not depend on the upstream ``geobench``
package. Loads samples directly from ``classification_v1.0/<dataset>/``
HDF5 files using the partition JSON files distributed alongside them.
"""

import io
import json
import pickle
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .base import BenchDataset

V1_ROOT = Path("data/classification_v1.0")


@dataclass
class BandStats:
    """Band statistics with type-safe access.

    Attributes:
        min: Minimum value.
        max: Maximum value.
        mean: Mean value.
        std: Standard deviation.
        median: Median value.
        percentile_0_1: 0.1th percentile.
        percentile_1: 1st percentile.
        percentile_2: 2nd percentile (interpolated if not available).
        percentile_5: 5th percentile.
        percentile_95: 95th percentile.
        percentile_98: 98th percentile (interpolated if not available).
        percentile_99: 99th percentile.
        percentile_99_9: 99.9th percentile.
    """

    min: float
    max: float
    mean: float
    std: float
    median: float
    percentile_0_1: float
    percentile_1: float
    percentile_2: float | None = None
    percentile_5: float | None = None
    percentile_95: float | None = None
    percentile_98: float | None = None
    percentile_99: float | None = None
    percentile_99_9: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        """Create BandStats from a dict, interpolating p2/p98 if missing."""
        if "percentile_2" not in d and "percentile_1" in d and "percentile_5" in d:
            d["percentile_2"] = d["percentile_1"] + 0.25 * (d["percentile_5"] - d["percentile_1"])
        if "percentile_98" not in d and "percentile_95" in d and "percentile_99" in d:
            d["percentile_98"] = d["percentile_95"] + 0.75 * (
                d["percentile_99"] - d["percentile_95"]
            )
        return cls(**d)


class GeoBenchv1(Dataset):
    """PyTorch Dataset for GeoBench V1 classification benchmarks.

    Args:
        root: Path to the GeoBench V1 collection (e.g. ``data/classification_v1.0``).
        dataset_name: Dataset name (e.g. ``"m-eurosat"``).
        split: Split name (``"train"``, ``"valid"``, or ``"test"``).
        partition: Partition name (e.g. ``"default"``, ``"0.01x_train"``).
        bands: Tuple of source band names (``"04 - Red"``, etc.) to load. If
            ``None``, loads all bands present in the first sample.
        transform: Optional callable applied to each sample dict.
        normalize: **Deprecated** — kept for API back-compat for one cycle.
            Always ignored: this class now emits raw float32 values.  Anything
            other than ``False``/``None``/``"none"`` triggers a
            :class:`DeprecationWarning`.  Per-channel normalization belongs on
            :class:`~torchgeo_bench.models.interface.BenchModel`.
    """

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: Literal["train", "valid", "test"],
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: object = None,
        normalize: bool | str | None = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.partition = partition
        self.transform = transform

        if normalize not in (None, False, "none"):
            warnings.warn(
                "GeoBenchv1.normalize is deprecated and ignored: this class emits "
                "raw values; per-channel normalization is now done by BenchModel.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.dataset_dir = self.root / dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        partition_file = self.dataset_dir / f"{partition}_partition.json"
        if not partition_file.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_file}")

        with open(partition_file) as f:
            partition_data = json.load(f)

        if split not in partition_data:
            raise ValueError(
                f"Split '{split}' not found in partition. Available: {list(partition_data.keys())}"
            )
        self.sample_ids = partition_data[split]

        if bands is None:
            sample_meta = self._load_sample_metadata(self.sample_ids[0])
            self.band_names: list[str] = list(sample_meta["bands_order"])
        else:
            self.band_names = list(bands)

    def _load_sample_metadata(self, sample_id: str) -> dict:
        """Load pickled metadata from HDF5 attributes."""
        sample_path = self.dataset_dir / f"{sample_id}.hdf5"
        with h5py.File(sample_path, "r") as f:
            pickle_str = f.attrs["pickle"]
            try:
                metadata = pickle.loads(eval(pickle_str))  # type: ignore[arg-type]
            except (ModuleNotFoundError, AttributeError):
                # Pickle references geobench module classes we don't need.
                # Use a restricted unpickler that stubs them out.
                class _RestrictedUnpickler(pickle.Unpickler):
                    def find_class(self, module: str, name: str) -> type:  # type: ignore[override]
                        if module == "geobench.dataset":
                            return type(name, (), {})
                        return super().find_class(module, name)

                metadata = _RestrictedUnpickler(io.BytesIO(eval(pickle_str))).load()  # type: ignore[arg-type]
        return metadata

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        sample_path = self.dataset_dir / f"{sample_id}.hdf5"

        metadata = self._load_sample_metadata(sample_id)
        label = metadata["label"]

        bands_data = []
        with h5py.File(sample_path, "r") as f:
            available_keys = list(f.keys())
            for band_name in self.band_names:
                if band_name in available_keys:
                    data = f[band_name][:]  # type: ignore[index]
                else:
                    # Temporal datasets (e.g. m-forestnet) suffix bands with dates.
                    matching = [k for k in available_keys if k.startswith(band_name)]
                    if not matching:
                        raise KeyError(
                            f"Band '{band_name}' not found in {sample_path}. "
                            f"Available: {available_keys[:5]}..."
                        )
                    data = f[matching[0]][:]  # type: ignore[index]
                bands_data.append(data)

        image = np.stack(bands_data, axis=0).astype(np.float32)

        image_t = torch.from_numpy(image)
        label_arr = np.asarray(label)
        label_t: torch.Tensor
        if label_arr.ndim > 0:
            label_t = torch.from_numpy(label_arr.astype(np.float32))
        else:
            label_t = torch.tensor(label_arr.item(), dtype=torch.long)

        sample: dict = {"image": image_t, "label": label_t, "sample_id": sample_id}
        if self.transform is not None:
            sample = self.transform(sample)  # type: ignore[misc]
        return sample

    @cached_property
    def band_stats(self) -> dict[str, BandStats]:
        """Load band statistics with caching."""
        band_stats_file = self.dataset_dir / "band_stats.json"
        if not band_stats_file.exists():
            return {}
        with open(band_stats_file) as f:
            stats_dict = json.load(f)
        return {name: BandStats.from_dict(stats) for name, stats in stats_dict.items()}


class _V1Dataset(BenchDataset):
    """Base class for every GeoBench V1 wrapper.

    Concrete subclasses just declare metadata (``name``, ``num_classes``,
    ``bands``, ``rgb_bands``, ``split_sizes``, ``multilabel``); ``get_dataset``
    is fully implemented here and dispatches to :class:`GeoBenchv1`.
    """

    supports_partitions = True

    @classmethod
    def data_root(cls) -> Path:
        return V1_ROOT

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a :class:`GeoBenchv1` for the given split (raw values)."""
        v1_split: Literal["train", "valid", "test"] = "valid" if split == "val" else split  # type: ignore[assignment]
        source_bands = tuple(spec.source_name for spec in self.select_band_specs(bands))
        return GeoBenchv1(
            root=self.data_root(),
            dataset_name=self.name,
            split=v1_split,
            partition=partition,
            bands=source_bands,
            transform=transform,
        )
