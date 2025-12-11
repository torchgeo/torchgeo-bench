"""Simple PyTorch Dataset for GeoBench classification datasets.

This provides a lightweight alternative to the geobench library that directly
accesses HDF5 files and partition JSON files.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class BandStats:
    """Band statistics with type-safe access.

    Attributes:
        min: Minimum value
        max: Maximum value
        mean: Mean value
        std: Standard deviation
        median: Median value
        percentile_0_1: 0.1th percentile
        percentile_1: 1st percentile
        percentile_2: 2nd percentile (interpolated if not available)
        percentile_5: 5th percentile
        percentile_95: 95th percentile
        percentile_98: 98th percentile (interpolated if not available)
        percentile_99: 99th percentile
        percentile_99_9: 99.9th percentile
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
    def from_dict(cls, d: dict) -> BandStats:
        """Create BandStats from dictionary.

        Args:
            d: Dictionary with band statistics

        Returns:
            BandStats instance with interpolated percentiles if missing
        """
        # Interpolate 2nd and 98th percentiles if not available
        if "percentile_2" not in d and "percentile_1" in d and "percentile_5" in d:
            # Linear interpolation: p2 is 1/4 of the way from p1 to p5
            d["percentile_2"] = d["percentile_1"] + 0.25 * (d["percentile_5"] - d["percentile_1"])
        if "percentile_98" not in d and "percentile_95" in d and "percentile_99" in d:
            # Linear interpolation: p98 is 3/4 of the way from p95 to p99
            d["percentile_98"] = d["percentile_95"] + 0.75 * (d["percentile_99"] - d["percentile_95"])
        return cls(**d)


class GeoBenchDataset(Dataset):
    """PyTorch Dataset for GeoBench classification benchmarks.

    Args:
        root: Path to GeoBench data directory (e.g., '/path/to/data/classification_v1.0')
        dataset_name: Dataset name (e.g., 'm-eurosat', 'm-forestnet')
        split: Split name ('train', 'valid', 'test')
        partition: Partition name (e.g., 'default', '0.01x_train', '0.10x_train')
        bands: Tuple of band names to load (e.g., ('red', 'green', 'blue')).
               If None, loads all bands. Band names are matched case-insensitively
               and can be short names like 'red' or full names like '04 - Red'.
        transform: Optional callable to transform the sample dict.
        normalize: If True, normalize using mean/std from band_stats.json.
                   If 'min_max', normalize to [0, 1] using min/max.
                   If 'percentile_2_98', normalize to [0, 1] using 2nd and 98th percentiles.
                   If False, no normalization (raw int16 values).

    Returns:
        Dictionary with keys:
            - 'image': torch.Tensor of shape (C, H, W) float32
            - 'label': torch.Tensor scalar (int64)
            - 'sample_id': str
    """

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: Literal["train", "valid", "test"],
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: object = None,
        normalize: bool | str = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.split = split
        self.partition = partition
        self.transform = transform
        self.normalize = normalize

        # Locate dataset directory
        self.dataset_dir = self.root / dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        # Load partition file
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

        # Determine band names to load
        if bands is None:
            # Load first sample to get all available bands
            sample_meta = self._load_sample_metadata(self.sample_ids[0])
            self.band_names = sample_meta["bands_order"]
        else:
            # Map requested bands to full names
            self.band_names = self._resolve_band_names(bands)

    def _resolve_band_names(self, requested_bands: tuple[str, ...]) -> list[str]:
        """Map short band names (e.g., 'red') to full names (e.g., '04 - Red').

        Note: For temporal datasets, returns base band names without date suffixes.
        The actual date-specific keys will be resolved per-sample in __getitem__.
        """
        # Load first sample to get available band names
        sample_meta = self._load_sample_metadata(self.sample_ids[0])
        available_bands = sample_meta["bands_order"]

        resolved = []
        for req in requested_bands:
            req_lower = req.lower()
            # Try exact match first
            if req in available_bands:
                resolved.append(req)
                continue

            # Try case-insensitive partial match with word boundaries
            # Priority: exact word match > starts with > contains
            found = False
            best_match = None
            best_score = -1

            for avail in available_bands:
                # Strip any date suffix for matching (temporal datasets)
                # E.g., '04 - Red_2013-01-01' -> '04 - Red'
                avail_base = avail.split("_")[0] if "_" in avail else avail
                avail_lower = avail_base.lower()

                if req_lower == avail_lower:
                    # Exact match (case-insensitive)
                    best_match = avail_base
                    best_score = 100
                    break
                elif req_lower in avail_lower:
                    # Check if it's a word boundary match (e.g., "red" matches "04 - Red" but not "FilteRed")
                    # Split by common delimiters and check if request matches any token
                    tokens = avail_lower.replace("-", " ").replace(".", " ").split()
                    if req_lower in tokens:
                        # Exact word match (highest priority for partial matches)
                        score = 50
                    elif any(token.startswith(req_lower) for token in tokens):
                        # Token starts with request
                        score = 30
                    else:
                        # Substring match (lowest priority)
                        score = 10

                    if score > best_score:
                        best_match = avail_base
                        best_score = score

            if best_match:
                resolved.append(best_match)
                found = True

            if not found:
                raise ValueError(f"Band '{req}' not found in available bands: {available_bands}")
        return resolved

    def _load_sample_metadata(self, sample_id: str) -> dict:
        """Load pickled metadata from HDF5 attributes."""
        sample_path = self.dataset_dir / f"{sample_id}.hdf5"
        with h5py.File(sample_path, "r") as f:
            # Pickle is stored as string in attrs, need to eval then unpickle
            pickle_str = f.attrs["pickle"]
            # Create a safe unpickler that doesn't require geobench module
            # We only need the label and bands_order from the metadata
            try:
                metadata = pickle.loads(eval(pickle_str))  # type: ignore[arg-type]
            except (ModuleNotFoundError, AttributeError):
                # Fallback: manually extract what we need without full unpickling
                # The pickle contains geobench classes we don't need to reconstruct
                import io

                # Use restricted unpickler to get basic dict structure
                class RestrictedUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):  # type: ignore[override]
                        # Allow basic types and geobench classes (will become stubs)
                        if module == "geobench.dataset":
                            # Return a dummy class that can be instantiated
                            return type(name, (), {})
                        return super().find_class(module, name)

                metadata = RestrictedUnpickler(io.BytesIO(eval(pickle_str))).load()  # type: ignore[arg-type]
        return metadata

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        sample_path = self.dataset_dir / f"{sample_id}.hdf5"

        # Load metadata
        metadata = self._load_sample_metadata(sample_id)
        label = metadata["label"]

        # Load band data
        bands_data = []
        with h5py.File(sample_path, "r") as f:
            # Get available keys in the HDF5 file
            available_keys = list(f.keys())

            for band_name in self.band_names:
                # Try exact match first
                if band_name in available_keys:
                    data = f[band_name][:]  # type: ignore[index]
                else:
                    # For temporal datasets (e.g., m-forestnet), bands may have date suffixes
                    # Look for keys that start with the band name
                    matching_keys = [k for k in available_keys if k.startswith(band_name)]
                    if matching_keys:
                        # Use the first matching key (typically there's only one date per sample)
                        data = f[matching_keys[0]][:]  # type: ignore[index]
                    else:
                        raise KeyError(
                            f"Band '{band_name}' not found in {sample_path}. "
                            f"Available keys: {available_keys[:5]}..."
                        )
                bands_data.append(data)

        # Stack to (C, H, W)
        image = np.stack(bands_data, axis=0).astype(np.float32)

        # Normalize
        if self.normalize:
            if self.normalize == "min_max":
                # Normalize to [0, 1] using min/max from band stats
                for i, band_name in enumerate(self.band_names):
                    if band_name in self.band_stats:
                        stats = self.band_stats[band_name]
                        image[i] = (image[i] - stats.min) / (stats.max - stats.min + 1e-8)
            elif self.normalize == "percentile_2_98":
                # Normalize to [0, 1] using 2nd and 98th percentiles, clipping outliers
                for i, band_name in enumerate(self.band_names):
                    if band_name in self.band_stats:
                        stats = self.band_stats[band_name]
                        if stats.percentile_2 is not None and stats.percentile_98 is not None:
                            p2, p98 = stats.percentile_2, stats.percentile_98
                            image[i] = np.clip((image[i] - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
            else:
                # Normalize using mean/std
                for i, band_name in enumerate(self.band_names):
                    if band_name in self.band_stats:
                        stats = self.band_stats[band_name]
                        image[i] = (image[i] - stats.mean) / (stats.std + 1e-8)

        # Convert to torch tensors
        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.long)

        sample = {"image": image, "label": label, "sample_id": sample_id}

        if self.transform is not None:
            sample = self.transform(sample)  # type: ignore[misc]

        return sample

    def get_num_classes(self) -> int:
        """Infer number of classes from all labels in the split."""
        labels = set()
        for sample_id in self.sample_ids:
            metadata = self._load_sample_metadata(sample_id)
            labels.add(metadata["label"])
        return len(labels)

    @cached_property
    def band_stats(self) -> dict[str, BandStats]:
        """Load band statistics with caching.

        Returns dict mapping band names to BandStats objects.
        Cached after first access to avoid repeated file I/O.

        Returns:
            Dictionary mapping band name to BandStats
        """
        band_stats_file = self.dataset_dir / "band_stats.json"
        if not band_stats_file.exists():
            return {}

        with open(band_stats_file) as f:
            stats_dict = json.load(f)

        # Convert to BandStats objects for type-safe access
        return {name: BandStats.from_dict(stats) for name, stats in stats_dict.items()}

    @classmethod
    def get_available_bands(cls, root: str | Path, dataset_name: str) -> list[str]:
        """Get list of available bands for a dataset without loading samples.

        Args:
            root: Path to classification_v1.0 directory
            dataset_name: Dataset name (e.g., 'm-eurosat')

        Returns:
            List of available band names

        Example:
            >>> bands = GeoBenchDataset.get_available_bands(
            ...     root='/path/to/data/classification_v1.0',
            ...     dataset_name='m-eurosat'
            ... )
            >>> print(bands)
            ['01 - Coastal aerosol', '02 - Blue', '03 - Green', ...]
        """
        dataset_dir = Path(root) / dataset_name
        band_stats_file = dataset_dir / "band_stats.json"

        if not band_stats_file.exists():
            raise FileNotFoundError(
                f"band_stats.json not found for dataset '{dataset_name}' at {band_stats_file}"
            )

        with open(band_stats_file) as f:
            stats_dict = json.load(f)

        return list(stats_dict.keys())


def get_geobench_dataset(
    root: str | Path,
    dataset_name: str,
    split: Literal["train", "valid", "test"],
    partition: str = "default",
    bands: tuple[str, ...] = ("red", "green", "blue"),
    normalize: bool | str = True,
    transform: object = None,
) -> GeoBenchDataset:
    """Factory function to create a GeoBenchDataset.

    Args:
        root: Path to classification_v1.0 directory
        dataset_name: e.g., 'm-eurosat', 'm-forestnet'
        split: 'train', 'valid', or 'test'
        partition: e.g., 'default', '0.01x_train', '0.10x_train'
        bands: Tuple of band names (e.g., ('red', 'green', 'blue'))
        normalize: True (mean/std), 'min_max' (0-1), or False (raw)
        transform: Optional transform callable

    Returns:
        GeoBenchDataset instance
    """
    return GeoBenchDataset(
        root=root,
        dataset_name=dataset_name,
        split=split,
        partition=partition,
        bands=bands,
        transform=transform,
        normalize=normalize,
    )
