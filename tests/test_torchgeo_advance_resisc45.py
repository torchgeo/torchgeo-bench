"""Mock-based tests for torchgeo ADVANCE and RESISC45 wrappers."""

import json
from pathlib import Path

import pytest
import torch

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.datasets.advance import ADVANCE as AdvanceBench


class MockAdvanceDataset:
    """Deterministic stand-in for ``torchgeo.datasets.ADVANCE``."""

    getitem_calls = 0

    def __init__(
        self,
        root: str | Path = "data",
        transforms=None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        del download, checksum
        self.root = Path(root)
        self.transforms = transforms
        self._classes = [f"class_{i:02d}" for i in range(13)]
        self.files: list[dict[str, str]] = []
        for i in range(5075):
            class_name = self._classes[i % len(self._classes)]
            image = self.root / "vision" / class_name / f"img_{i:05d}.jpg"
            audio = self.root / "sound" / class_name / f"aud_{i:05d}.wav"
            self.files.append({"image": str(image), "audio": str(audio), "cls": class_name})

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict:
        type(self).getitem_calls += 1
        base = index % 251
        image = torch.stack(
            [
                torch.full((8, 8), base, dtype=torch.uint8),
                torch.full((8, 8), base + 1, dtype=torch.uint8),
                torch.full((8, 8), base + 2, dtype=torch.uint8),
            ],
            dim=0,
        )
        sample = {
            "image": image,
            "audio": torch.tensor([float(index)], dtype=torch.float32),
            "label": torch.tensor(index % 13, dtype=torch.long),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class MockResisc45Dataset:
    """Deterministic stand-in for ``torchgeo.datasets.RESISC45``."""

    SPLIT_LENGTHS = {"train": 18900, "val": 6300, "test": 6300}

    def __init__(
        self,
        root: str | Path = "data",
        split: str = "train",
        transforms=None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        del root, download, checksum
        self.split = split
        self.transforms = transforms

    def __len__(self) -> int:
        return self.SPLIT_LENGTHS[self.split]

    def __getitem__(self, index: int) -> dict:
        base = index % 255
        image = torch.stack(
            [
                torch.full((8, 8), base, dtype=torch.uint8),
                torch.full((8, 8), min(base + 3, 255), dtype=torch.uint8),
                torch.full((8, 8), min(base + 6, 255), dtype=torch.uint8),
            ],
            dim=0,
        )
        sample = {"image": image, "label": torch.tensor(index % 45, dtype=torch.long)}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


@pytest.fixture
def patch_torchgeo_datasets(monkeypatch, tmp_path):
    """Patch torchgeo dataset classes for ADVANCE and RESISC45 wrappers."""
    from torchgeo_bench.datasets import advance as advance_module
    from torchgeo_bench.datasets import resisc45 as resisc45_module

    monkeypatch.setattr(advance_module, "TGADVANCE", MockAdvanceDataset)
    monkeypatch.setattr(resisc45_module, "TGRESISC45", MockResisc45Dataset)
    monkeypatch.setattr(
        AdvanceBench,
        "data_root",
        classmethod(lambda cls: tmp_path / "advance"),
    )


def test_registry_contains_new_datasets():
    """Dataset registry should expose both new torchgeo wrappers."""
    assert get_bench_dataset_class("advance").name == "advance"
    assert get_bench_dataset_class("resisc45").name == "resisc45"


def test_get_datasets_return_val_for_advance_and_resisc45(patch_torchgeo_datasets):
    """``get_datasets(..., return_val=True)`` should work for both datasets."""
    del patch_torchgeo_datasets

    advance_ds, advance_train, advance_val, advance_test = get_datasets(
        dataset_name="advance",
        return_val=True,
        batch_size=16,
        num_workers=0,
    )
    assert len(advance_ds) == 3045
    assert len(advance_val.dataset) == 1015
    assert len(advance_test.dataset) == 1015
    advance_batch = next(iter(advance_train))
    assert advance_batch["image"].shape[1] == 3
    assert "audio" not in advance_batch

    resisc_ds, resisc_train, resisc_val, resisc_test = get_datasets(
        dataset_name="resisc45",
        return_val=True,
        batch_size=16,
        num_workers=0,
    )
    assert len(resisc_ds) == 18900
    assert len(resisc_val.dataset) == 6300
    assert len(resisc_test.dataset) == 6300
    resisc_batch = next(iter(resisc_train))
    assert resisc_batch["image"].shape[1] == 3


def test_advance_creates_and_reuses_split_and_stats_cache(patch_torchgeo_datasets, tmp_path):
    """ADVANCE split and band-stats files should be persisted and reused."""
    del patch_torchgeo_datasets
    MockAdvanceDataset.getitem_calls = 0

    bench_first = AdvanceBench()
    split_file = tmp_path / "advance" / "torchgeo_bench_split_seed42_v1.json"
    stats_file = tmp_path / "advance" / "torchgeo_bench_band_stats_seed42_v1.json"
    assert split_file.exists()
    assert stats_file.exists()

    with split_file.open("r", encoding="utf-8") as file:
        split_payload_first = json.load(file)
    assert len(split_payload_first["train"]) == 3045
    assert len(split_payload_first["val"]) == 1015
    assert len(split_payload_first["test"]) == 1015

    first_getitem_calls = MockAdvanceDataset.getitem_calls
    assert first_getitem_calls == 3045

    bench_second = AdvanceBench()
    with split_file.open("r", encoding="utf-8") as file:
        split_payload_second = json.load(file)

    assert split_payload_second == split_payload_first
    assert MockAdvanceDataset.getitem_calls == first_getitem_calls
    assert [spec.name for spec in bench_second.bands] == ["red", "green", "blue"]
    assert all(spec.max >= spec.min for spec in bench_second.bands)
    assert bench_first.bands == bench_second.bands


def test_advance_wrapper_outputs_image_and_label_only(patch_torchgeo_datasets):
    """ADVANCE wrapper should strip the upstream ``audio`` key from samples."""
    del patch_torchgeo_datasets
    bench = AdvanceBench()

    for split, expected_length in {"train": 3045, "val": 1015, "test": 1015}.items():
        ds = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
        assert len(ds) == expected_length
        sample = ds[0]
        assert set(sample) == {"image", "label"}
        assert sample["image"].shape[0] == 3


def test_resisc45_split_loading_works(patch_torchgeo_datasets):
    """RESISC45 wrapper should expose train/val/test with expected lengths."""
    del patch_torchgeo_datasets
    bench_cls = get_bench_dataset_class("resisc45")
    bench = bench_cls()

    assert len(bench.get_dataset("train", bands=tuple(bench.rgb_bands))) == 18900
    assert len(bench.get_dataset("val", bands=tuple(bench.rgb_bands))) == 6300
    assert len(bench.get_dataset("test", bands=tuple(bench.rgb_bands))) == 6300
