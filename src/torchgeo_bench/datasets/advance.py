"""ADVANCE (torchgeo) benchmark dataset wrapper with fixed splits."""

import json
import os
import random
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import ADVANCE as TGADVANCE
from torchgeo.datasets.errors import DatasetNotFoundError

from .base import BandSpec, BenchDataset

SPLIT_SEED = 42
SPLIT_VERSION = "v2"
SPLIT_FILENAME = f"torchgeo_bench_split_seed{SPLIT_SEED}_{SPLIT_VERSION}.json"
STATS_FILENAME = f"torchgeo_bench_band_stats_seed{SPLIT_SEED}_{SPLIT_VERSION}.json"
EXPECTED_SPLIT_SIZES: dict[str, int] = {"train": 3045, "val": 1015, "test": 1015}
SPLIT_RATIOS: dict[str, float] = {"train": 0.6, "val": 0.2, "test": 0.2}
RGB_NAMES = ("red", "green", "blue")
RGB_WAVELENGTHS = {"red": 0.665, "green": 0.56, "blue": 0.49}


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically write JSON data using a temp file and ``replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    tmp_path.replace(path)


class _AdvanceDatasetView(Dataset):
    """Split and channel view over the torchgeo ADVANCE dataset."""

    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        channel_indices: list[int],
        transform: Callable | None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.channel_indices = channel_indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        sample = self.base_dataset[self.indices[index]]
        image = sample["image"][self.channel_indices].to(dtype=torch.float32)
        out = {"image": image, "label": sample["label"]}
        if self.transform is not None:
            out = self.transform(out)
        return out


class ADVANCE(BenchDataset):
    """ADVANCE audio-visual scene dataset with torchgeo-bench fixed split.

    Split logic follows the class-stratified ratios from the official repo
    (train/test 80/20) and extends it to 60/20/20 for train/val/test.
    Reference: https://github.com/DTaoo/Multimodal-Aerial-Scene-Recognition/
    blob/b5345f5e1b4b490b2a1ab1317236a9fe81bef761/resnet-image/utils.py
    """

    name = "advance"
    task = "classification"
    num_classes = 13
    multilabel = False
    rgb_bands = list(RGB_NAMES)
    split_sizes = dict(EXPECTED_SPLIT_SIZES)
    supports_partitions = False

    # Placeholder values are replaced per-instance from cached train-split stats.
    bands = [
        BandSpec("aerial", "red", "red", mean=0.0, std=1.0, min=0.0, max=255.0, wavelength_um=0.665),
        BandSpec(
            "aerial",
            "green",
            "green",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=255.0,
            wavelength_um=0.56,
        ),
        BandSpec("aerial", "blue", "blue", mean=0.0, std=1.0, min=0.0, max=255.0, wavelength_um=0.49),
    ]

    def __init__(self) -> None:
        self._root = self.data_root()
        self._dataset: Dataset | None = None
        self._path_to_index: dict[str, int] = {}
        self._split_paths: dict[str, list[str]] = {}
        self._init_error: DatasetNotFoundError | None = None
        self.bands = list(type(self).bands)
        try:
            self._initialize_runtime_state()
        except DatasetNotFoundError as exc:
            self._init_error = exc

    @classmethod
    def data_root(cls) -> Path:
        return Path("data/advance")

    @property
    def _split_file(self) -> Path:
        return self._root / SPLIT_FILENAME

    @property
    def _stats_file(self) -> Path:
        return self._root / STATS_FILENAME

    def _initialize_runtime_state(self) -> None:
        if self._dataset is not None:
            return
        self._dataset = TGADVANCE(
            root=str(self._root),
            transforms=None,
            download=True,
        )
        self._path_to_index = self._build_path_to_index()
        self._split_paths = self._load_or_create_split_file()
        self._validate_split_paths(self._split_paths)
        self.bands = self._load_or_compute_band_stats()
        self._init_error = None

    def _ensure_initialized(self) -> None:
        if self._dataset is not None:
            return
        if self._init_error is not None:
            raise self._init_error
        self._initialize_runtime_state()

    def _relative_vision_path(self, image_path: str) -> str:
        path = Path(image_path)
        if path.is_absolute():
            rel = path.relative_to(self._root.resolve())
        else:
            rel = path.relative_to(self._root) if path.parts[: len(self._root.parts)] == self._root.parts else path
        if not rel.parts or rel.parts[0] != "vision":
            raise ValueError(f"Unexpected ADVANCE image path outside vision/: {image_path}")
        return rel.as_posix()

    def _build_path_to_index(self) -> dict[str, int]:
        assert self._dataset is not None
        mapping: dict[str, int] = {}
        for idx, file_entry in enumerate(self._dataset.files):
            rel_path = self._relative_vision_path(file_entry["image"])
            if rel_path in mapping:
                raise ValueError(f"Duplicate ADVANCE image path detected: {rel_path}")
            mapping[rel_path] = idx
        return mapping

    def _label_for_index(self, index: int) -> int:
        assert self._dataset is not None
        file_entry = self._dataset.files[index]
        label = file_entry.get("label")
        if label is None:
            sample = self._dataset[index]
            label = sample["label"]
        return int(label)

    def _class_stratified_paths(self) -> dict[int, list[str]]:
        assert self._dataset is not None
        label_to_paths: dict[int, list[str]] = {}
        for idx, file_entry in enumerate(self._dataset.files):
            rel_path = self._relative_vision_path(file_entry["image"])
            label = self._label_for_index(idx)
            label_to_paths.setdefault(label, []).append(rel_path)
        return label_to_paths

    def _split_counts_for_class(self, count: int) -> dict[str, int]:
        train = int(round(count * SPLIT_RATIOS["train"]))
        val = int(round(count * SPLIT_RATIOS["val"]))
        test = count - train - val
        if train < 0 or val < 0 or test < 0:
            raise ValueError(f"Invalid split allocation for class size {count}.")
        return {"train": train, "val": val, "test": test}

    def _rebalance_split_totals(
        self,
        class_splits: dict[int, dict[str, list[str]]],
        totals: dict[str, int],
    ) -> None:
        targets = dict(EXPECTED_SPLIT_SIZES)
        splits = ["train", "val", "test"]
        while any(totals[split] != targets[split] for split in splits):
            surplus_splits = [s for s in splits if totals[s] > targets[s]]
            deficit_splits = [s for s in splits if totals[s] < targets[s]]
            if not surplus_splits or not deficit_splits:
                break

            surplus_splits.sort(key=lambda s: (targets[s] - totals[s]))
            deficit_splits.sort(key=lambda s: (targets[s] - totals[s]), reverse=True)
            from_split = surplus_splits[0]
            to_split = deficit_splits[0]

            moved = False
            for label in sorted(class_splits):
                if class_splits[label][from_split]:
                    sample = class_splits[label][from_split].pop()
                    class_splits[label][to_split].append(sample)
                    totals[from_split] -= 1
                    totals[to_split] += 1
                    moved = True
                    break
            if not moved:
                raise ValueError(
                    "Unable to rebalance ADVANCE splits to expected sizes."
                )

    def _make_split_payload(self) -> dict:
        all_paths = sorted(self._path_to_index)
        total_expected = sum(EXPECTED_SPLIT_SIZES.values())
        if len(all_paths) != total_expected:
            raise ValueError(
                "ADVANCE sample count mismatch: expected "
                f"{total_expected}, found {len(all_paths)}."
            )

        rng = random.Random(SPLIT_SEED)
        label_to_paths = self._class_stratified_paths()
        class_splits: dict[int, dict[str, list[str]]] = {}

        for label in sorted(label_to_paths):
            paths = list(label_to_paths[label])
            rng.shuffle(paths)
            counts = self._split_counts_for_class(len(paths))
            train_end = counts["train"]
            val_end = train_end + counts["val"]
            class_splits[label] = {
                "train": paths[:train_end],
                "val": paths[train_end:val_end],
                "test": paths[val_end:],
            }

        totals = {
            split: sum(len(class_splits[label][split]) for label in class_splits)
            for split in ("train", "val", "test")
        }
        if totals != EXPECTED_SPLIT_SIZES:
            self._rebalance_split_totals(class_splits, totals)
        if totals != EXPECTED_SPLIT_SIZES:
            raise ValueError(
                "ADVANCE split sizes do not match expected totals after rebalancing."
            )

        train_paths: list[str] = []
        val_paths: list[str] = []
        test_paths: list[str] = []
        for label in sorted(class_splits):
            train_paths.extend(class_splits[label]["train"])
            val_paths.extend(class_splits[label]["val"])
            test_paths.extend(class_splits[label]["test"])

        rng.shuffle(train_paths)
        rng.shuffle(val_paths)
        rng.shuffle(test_paths)

        return {
            "seed": SPLIT_SEED,
            "version": SPLIT_VERSION,
            "ratios": SPLIT_RATIOS,
            "train": train_paths,
            "val": val_paths,
            "test": test_paths,
        }

    def _validate_split_payload(self, payload: dict) -> dict[str, list[str]]:
        split_paths: dict[str, list[str]] = {}
        for split, expected_size in EXPECTED_SPLIT_SIZES.items():
            paths = payload.get(split)
            if not isinstance(paths, list):
                raise ValueError(f"ADVANCE split file missing list for split {split!r}.")
            if len(paths) != expected_size:
                raise ValueError(
                    f"ADVANCE split {split!r} has {len(paths)} samples; expected {expected_size}."
                )
            split_paths[split] = [str(path) for path in paths]

        merged = split_paths["train"] + split_paths["val"] + split_paths["test"]
        if len(set(merged)) != len(merged):
            raise ValueError("ADVANCE split file contains duplicate paths across splits.")
        return split_paths

    def _load_or_create_split_file(self) -> dict[str, list[str]]:
        if self._split_file.exists():
            with self._split_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        else:
            payload = self._make_split_payload()
            _atomic_write_json(self._split_file, payload)

        return self._validate_split_payload(payload)

    def _validate_split_paths(self, split_paths: dict[str, list[str]]) -> None:
        for split, paths in split_paths.items():
            missing = [path for path in paths if path not in self._path_to_index]
            if missing:
                raise ValueError(
                    f"ADVANCE split {split!r} references missing images, e.g. {missing[0]!r}."
                )

    def _compute_stats_for_indices(self, indices: list[int]) -> dict[str, dict[str, float]]:
        assert self._dataset is not None
        sum_vals = torch.zeros(3, dtype=torch.float64)
        sum_sq_vals = torch.zeros(3, dtype=torch.float64)
        min_vals = torch.full((3,), float("inf"), dtype=torch.float64)
        max_vals = torch.full((3,), float("-inf"), dtype=torch.float64)
        n_pixels = 0

        for index in indices:
            sample = self._dataset[index]
            image = sample["image"].to(dtype=torch.float64)
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"Expected ADVANCE image shape (3,H,W), got {tuple(image.shape)}")

            flat = image.view(3, -1)
            sum_vals += flat.sum(dim=1)
            sum_sq_vals += (flat * flat).sum(dim=1)
            min_vals = torch.minimum(min_vals, flat.min(dim=1).values)
            max_vals = torch.maximum(max_vals, flat.max(dim=1).values)
            n_pixels += int(flat.shape[1])

        if n_pixels == 0:
            raise ValueError("Cannot compute ADVANCE stats from empty split.")

        mean_vals = sum_vals / n_pixels
        var_vals = torch.clamp(sum_sq_vals / n_pixels - mean_vals * mean_vals, min=0.0)
        std_vals = torch.sqrt(var_vals)

        stats: dict[str, dict[str, float]] = {}
        for i, name in enumerate(RGB_NAMES):
            stats[name] = {
                "mean": float(mean_vals[i].item()),
                "std": float(std_vals[i].item()),
                "min": float(min_vals[i].item()),
                "max": float(max_vals[i].item()),
            }
        return stats

    def _band_specs_from_stats(self, stats: dict[str, dict[str, float]]) -> list[BandSpec]:
        specs: list[BandSpec] = []
        for name in RGB_NAMES:
            if name not in stats:
                raise ValueError(f"ADVANCE stats file missing band {name!r}.")
            band_stats = stats[name]
            specs.append(
                BandSpec(
                    sensor="aerial",
                    name=name,
                    source_name=name,
                    mean=float(band_stats["mean"]),
                    std=float(band_stats["std"]),
                    min=float(band_stats["min"]),
                    max=float(band_stats["max"]),
                    wavelength_um=RGB_WAVELENGTHS[name],
                )
            )
        return specs

    def _load_or_compute_band_stats(self) -> list[BandSpec]:
        if self._stats_file.exists():
            with self._stats_file.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            stats = payload.get("bands")
            if not isinstance(stats, dict):
                raise ValueError("ADVANCE stats cache missing 'bands' object.")
            return self._band_specs_from_stats(stats)

        train_indices = [self._path_to_index[path] for path in self._split_paths["train"]]
        stats = self._compute_stats_for_indices(train_indices)
        payload = {
            "seed": SPLIT_SEED,
            "version": SPLIT_VERSION,
            "bands": stats,
        }
        _atomic_write_json(self._stats_file, payload)
        return self._band_specs_from_stats(stats)

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        del partition
        self._ensure_initialized()
        assert self._dataset is not None
        if split not in self.split_sizes:
            raise ValueError(f"Unsupported split: {split!r}. Expected one of {sorted(self.split_sizes)}")

        channel_lookup = {spec.name: idx for idx, spec in enumerate(self.bands)}
        selected_specs = self.select_band_specs(bands)
        selected_channels = [channel_lookup[spec.name] for spec in selected_specs]
        split_indices = [self._path_to_index[path] for path in self._split_paths[split]]
        return _AdvanceDatasetView(self._dataset, split_indices, selected_channels, transform)
