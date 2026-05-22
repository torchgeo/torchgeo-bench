"""SEN12MS and SEN12MS-CR dataset wrappers."""

import logging
import pickle
import random
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
from torchgeo.datasets.errors import DatasetNotFoundError

from .base import BandSpec, BenchDataset

logger = logging.getLogger(__name__)

_IGBP17_TO_10: dict[int, int] = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 1,
    6: 1,
    7: 2,
    8: 2,
    9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 5,
    14: 7,
    15: 8,
    16: 9,
}


class _SEN12MSView(Dataset):
    """Split view over SEN12MS file paths with channel selection."""

    def __init__(
        self,
        items: list[tuple[Path, int]],
        channel_indices: list[int],
        transform: Callable | None,
    ) -> None:
        self.items = items
        self.channel_indices = channel_indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        path, label = self.items[index]
        indexes = [i + 1 for i in self.channel_indices]
        with rasterio.open(path) as src:
            image = src.read(indexes=indexes).astype(np.float32)
        image_t = torch.from_numpy(image)
        out = {"image": image_t, "label": int(label)}
        if self.transform is not None:
            out = self.transform(out)
        return out


class _SEN12MSBase(BenchDataset):
    """Base wrapper for clean SEN12MS and cloud-binned SEN12MS-CR splits."""

    task = "classification"
    num_classes = 10
    multilabel = False
    supports_partitions = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes: dict[str, int] = {}
    prior_results_alias: str | None = None
    _cloud_bin: tuple[float, float] | None = None
    train_subset: int | None = 50_000

    # fmt: off
    bands = [
        BandSpec("s2", "coastal_aerosol", "B01", mean=1465.28, std=752.72, min=0, max=10000, wavelength_um=0.443),
        BandSpec("s2", "blue", "B02", mean=1230.45, std=748.00, min=0, max=10000, wavelength_um=0.490),
        BandSpec("s2", "green", "B03", mean=1141.88, std=746.68, min=0, max=10000, wavelength_um=0.560),
        BandSpec("s2", "red", "B04", mean=1144.56, std=967.37, min=0, max=10000, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "B05", mean=1356.36, std=953.73, min=0, max=10000, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "B06", mean=1941.11, std=990.37, min=0, max=10000, wavelength_um=0.740),
        BandSpec("s2", "red_edge_3", "B07", mean=2220.79, std=1086.71, min=0, max=10000, wavelength_um=0.783),
        BandSpec("s2", "nir", "B08", mean=2163.92, std=1061.98, min=0, max=10000, wavelength_um=0.842),
        BandSpec("s2", "water_vapour", "B09", mean=2418.99, std=1140.05, min=0, max=10000, wavelength_um=0.945),
        BandSpec("s2", "swir_cirrus", "B10", mean=792.98, std=584.31, min=0, max=10000, wavelength_um=1.375),
        BandSpec("s2", "swir_1", "B11", mean=23.99, std=34.17, min=0, max=10000, wavelength_um=1.610),
        BandSpec("s2", "swir_2", "B12", mean=2005.35, std=1138.48, min=0, max=10000, wavelength_um=2.190),
        BandSpec("s2", "red_edge_4", "B8A", mean=1358.41, std=997.34, min=0, max=10000, wavelength_um=0.865),
    ]
    # fmt: on

    def __init__(self) -> None:
        root = self.data_root()
        self._check_data_present(root, require_cloud=self._cloud_bin is not None)

        self._train_samples = self._filter_present(
            self._load_split(root / "train_list.pkl"), root, "s2", "train"
        )
        if self.train_subset is not None and self.train_subset < len(self._train_samples):
            rng = random.Random(0)
            self._train_samples = rng.sample(self._train_samples, self.train_subset)
        self._val_samples = self._filter_present(
            self._load_split(root / "val_list.pkl"), root, "s2", "val"
        )
        self._labels = self._load_labels(root / "IGBP_probability_labels.pkl")

        if self._cloud_bin is None:
            self._test_samples = self._filter_present(
                self._load_split(root / "test_list.pkl"), root, "s2", "test"
            )
        else:
            coverage = self._load_coverage(root / "cloud_coverage.pkl")
            all_cloudy = self._load_split(root / "test_list_cloudy.pkl")
            self._test_samples = []
            for sample in all_cloudy:
                sample_id = self._sample_id(sample)
                value = self._coverage_for_sample(sample_id, coverage)
                if self._in_cloud_bin(value):
                    self._test_samples.append(sample)

        self.split_sizes = {
            "train": len(self._train_samples),
            "val": len(self._val_samples),
            "test": len(self._test_samples),
        }

    @classmethod
    def data_root(cls) -> Path:
        return Path(__file__).resolve().parents[3] / "data/sen12ms_cr"

    @classmethod
    def _filter_present(
        cls,
        samples: list[dict[str, object]],
        root: Path,
        image_key: str,
        split: str,
    ) -> list[dict[str, object]]:
        return samples

    @classmethod
    def _check_data_present(cls, root: Path, *, require_cloud: bool) -> None:
        required = {
            "train_list.pkl",
            "val_list.pkl",
            "test_list.pkl",
            "IGBP_probability_labels.pkl",
        }
        if require_cloud:
            required.update({"test_list_cloudy.pkl", "cloud_coverage.pkl"})
        missing = sorted(file_name for file_name in required if not (root / file_name).exists())
        if missing or not root.exists() or not any(root.glob("ROIs*")):
            raise DatasetNotFoundError(
                f"SEN12MS-CR data not found at {root}. Missing: {missing}. "
                "Run scripts/download_sen12ms_cr.sh to download the data, then run "
                "scripts/compute_sen12ms_cloud_coverage.py to generate cloud_coverage.pkl."
            )

    @staticmethod
    def _load_split(path: Path) -> list[dict[str, object]]:
        with path.open("rb") as file:
            raw = pickle.load(file)
        samples = list(raw)
        for sample in samples:
            if not isinstance(sample, dict):
                raise ValueError(f"Invalid split entry in {path}: expected dict, got {type(sample)!r}")
        return samples

    @staticmethod
    def _load_labels(path: Path) -> dict[str, int]:
        with path.open("rb") as file:
            raw = pickle.load(file)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid labels payload at {path}: expected dict, got {type(raw)!r}")

        labels: dict[str, int] = {}
        for key, value in raw.items():
            probs = np.asarray(value).reshape(-1)
            if probs.size == 0:
                raise ValueError(f"Empty label probabilities for sample_id={key!r}.")
            cls_idx = int(np.argmax(probs))
            if probs.size == 17:
                cls_idx = _IGBP17_TO_10[cls_idx]
            elif probs.size != 10:
                raise ValueError(
                    f"Unexpected label probability length={probs.size} for sample_id={key!r}."
                )
            labels[str(key).removesuffix(".tif")] = cls_idx
        return labels

    @staticmethod
    def _load_coverage(path: Path) -> dict[str, float]:
        with path.open("rb") as file:
            raw = pickle.load(file)
        if not isinstance(raw, dict):
            raise ValueError(f"Invalid cloud coverage payload at {path}: expected dict.")
        return {str(k): float(v) for k, v in raw.items()}

    @staticmethod
    def _sample_id(sample: dict[str, object]) -> str:
        if "id" not in sample:
            raise ValueError(f"Split sample missing 'id' field: {sample}")
        return str(sample["id"])

    @staticmethod
    def _coverage_for_sample(sample_id: str, coverage: dict[str, float]) -> float:
        if sample_id not in coverage:
            raise ValueError(f"Missing cloud coverage for sample_id={sample_id}.")
        return coverage[sample_id]

    def _in_cloud_bin(self, value: float) -> bool:
        if self._cloud_bin is None:
            return True
        lo, hi = self._cloud_bin
        if hi == 100.0:
            return lo <= value <= hi
        return lo <= value < hi

    def _label_for_sample(self, sample_id: str) -> int:
        if sample_id not in self._labels:
            raise ValueError(f"Missing label for sample_id={sample_id}.")
        return int(self._labels[sample_id])

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        del partition
        split_map = {
            "train": self._train_samples,
            "val": self._val_samples,
            "test": self._test_samples,
        }
        if split not in split_map:
            raise ValueError(f"Unknown split {split!r}. Expected 'train', 'val', or 'test'.")
        samples = split_map[split]

        channel_lookup = {spec.name: idx for idx, spec in enumerate(self.bands)}
        selected_specs = self.select_band_specs(bands)
        channel_indices = [channel_lookup[spec.name] for spec in selected_specs]

        use_cloudy = self._cloud_bin is not None and split == "test"
        image_key = "s2_cloudy" if use_cloudy else "s2"
        root = self.data_root()

        items: list[tuple[Path, int]] = []
        for sample in samples:
            sample_id = self._sample_id(sample)
            label = self._label_for_sample(sample_id)
            items.append((root / sample[image_key], label))

        return _SEN12MSView(items, channel_indices, transform)


class SEN12MS(_SEN12MSBase):
    """Clean SEN12MS split wrapper."""

    name = "sen12ms"
    _cloud_bin = None


class SEN12MSCRC1(_SEN12MSBase):
    """SEN12MS-CR split with 0-20% cloud coverage."""

    name = "sen12ms_cr_c1"
    prior_results_alias = "sen12ms"
    _cloud_bin = (0.0, 20.0)


class SEN12MSCRC2(_SEN12MSBase):
    """SEN12MS-CR split with 20-40% cloud coverage."""

    name = "sen12ms_cr_c2"
    prior_results_alias = "sen12ms"
    _cloud_bin = (20.0, 40.0)


class SEN12MSCRC3(_SEN12MSBase):
    """SEN12MS-CR split with 40-60% cloud coverage."""

    name = "sen12ms_cr_c3"
    prior_results_alias = "sen12ms"
    _cloud_bin = (40.0, 60.0)


class SEN12MSCRC4(_SEN12MSBase):
    """SEN12MS-CR split with 60-80% cloud coverage."""

    name = "sen12ms_cr_c4"
    prior_results_alias = "sen12ms"
    _cloud_bin = (60.0, 80.0)


