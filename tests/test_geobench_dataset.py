"""Validate published GeoBench V1 samples; local imagery is optional."""

from itertools import pairwise

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tests.support.data import require_dataset_data
from torchgeo_bench.datasets import load_split

pytestmark = pytest.mark.slow


@pytest.fixture
def eurosat(small_partition: str) -> Dataset:
    require_dataset_data("m-eurosat")
    return load_split("m-eurosat", "train", partition=small_partition).dataset


@pytest.mark.parametrize(
    "dataset_name",
    ["m-eurosat", "m-forestnet", "m-so2sat", "m-pv4ger", "m-brick-kiln", "m-bigearthnet"],
)
def test_published_rgb_sample(dataset_name: str, small_partition: str) -> None:
    require_dataset_data(dataset_name)
    loaded = load_split(dataset_name, "train", partition=small_partition)
    dataset = loaded.dataset

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample["image"].ndim == 3
    assert sample["image"].shape[0] == 3
    assert sample["image"].dtype == torch.float32
    assert isinstance(sample["sample_id"], str)
    if loaded.multilabel:
        assert sample["label"].shape == (loaded.num_classes,)
        assert sample["label"].dtype == torch.float32
        assert ((sample["label"] == 0) | (sample["label"] == 1)).all()
    else:
        assert sample["label"].ndim == 0
        assert sample["label"].dtype == torch.long
        assert 0 <= sample["label"].item() < loaded.num_classes


def test_raw_sensor_values_are_not_normalized(eurosat: Dataset) -> None:
    image = eurosat[0]["image"]
    # A maximum above 100 distinguishes raw S2 counts from normalized inputs.
    assert image.max() > 100.0


def test_dataloader_preserves_sample_identity(eurosat: Dataset) -> None:
    loader = DataLoader(eurosat, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    assert batch["image"].shape[:2] == (4, 3)
    assert batch["label"].shape == (4,)
    assert batch["sample_id"] == [eurosat[index]["sample_id"] for index in range(4)]


def test_all_bands(small_partition: str) -> None:
    require_dataset_data("m-eurosat")
    dataset = load_split("m-eurosat", "train", partition=small_partition, bands="all").dataset
    assert dataset[0]["image"].shape[0] == 13


def test_partition_size_ordering() -> None:
    require_dataset_data("m-eurosat")
    partitions = ["0.01x_train", "0.02x_train", "0.05x_train", "0.10x_train"]
    sizes = [
        len(load_split("m-eurosat", "train", partition=partition).dataset)
        for partition in partitions
    ]
    assert all(left < right for left, right in pairwise(sizes))
