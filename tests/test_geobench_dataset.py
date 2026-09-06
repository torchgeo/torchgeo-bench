"""Tests for the published GeoBench V1 JSON shards.

These tests verify that the V1 readers can load all available
GeoBench V1 datasets with different partitions, splits, and normalization
methods. They access band data via the per-dataset wrapper's
``BenchDataset.get_dataset()``, which translates short canonical band names
(``"red"``, ``"green"``, etc.) to upstream source names like
``"04 - Red"`` for us.
"""

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.datasets._v1_webdataset import GeoBenchv1Sharded

# Source names used when bypassing the wrapper.
EUROSAT_RGB_SOURCE_BANDS = ("04 - Red", "03 - Green", "02 - Blue")


@pytest.mark.slow
class TestGeoBenchDatasetBasics:
    """Basic functionality tests for the sharded reader."""

    def test_dataset_initialization(self, geobench_root):
        """Test that dataset can be initialized."""
        dataset = GeoBenchv1Sharded(
            root=geobench_root,
            dataset_name="m-eurosat",
            split="train",
            partition="default",
            bands=EUROSAT_RGB_SOURCE_BANDS,
        )
        assert len(dataset) > 0
        assert dataset.dataset_dir.name == "m-eurosat"

    def test_get_item(self, geobench_root):
        """Test that __getitem__ returns correct format."""
        dataset = GeoBenchv1Sharded(
            root=geobench_root,
            dataset_name="m-eurosat",
            split="train",
            partition="default",
            bands=EUROSAT_RGB_SOURCE_BANDS,
        )
        sample = dataset[0]

        assert "image" in sample
        assert "label" in sample
        assert "sample_id" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)
        assert isinstance(sample["sample_id"], str)
        assert sample["image"].dim() == 3
        assert sample["image"].shape[0] == 3
        assert sample["label"].dim() == 0


@pytest.mark.slow
class TestAllDatasets:
    """Test all available datasets with small partition (via the wrapper)."""

    @pytest.mark.parametrize(
        "dataset_name",
        ["m-eurosat", "m-forestnet", "m-so2sat", "m-pv4ger", "m-brick-kiln"],
    )
    def test_dataset_loads_small_partition(self, geobench_root, dataset_name, small_partition):
        """Each dataset can be loaded with the 0.01x_train partition."""
        if not (Path(geobench_root) / dataset_name).exists():
            pytest.skip(f"{dataset_name} data not supplied")
        bench = get_bench_dataset_class(dataset_name)()
        dataset = bench.get_dataset(
            "train",
            partition=small_partition,
            bands=tuple(bench.rgb_bands),
        )

        assert len(dataset) > 0, f"{dataset_name} has no samples"
        sample = dataset[0]
        assert sample["image"].shape[0] == 3, f"{dataset_name}: expected 3 (RGB) channels"
        assert sample["image"].dtype == torch.float32, f"{dataset_name}: expected float32"
        assert sample["label"].dtype == torch.long, f"{dataset_name}: expected int64 label"

        expected_classes = bench.num_classes
        assert 0 <= sample["label"].item() < expected_classes, (
            f"{dataset_name}: label out of range [0, {expected_classes})"
        )


@pytest.mark.slow
class TestRawEmission:
    """Datasets always emit raw float32 values; normalization moved to BenchModel."""

    def test_raw_pixel_range(self, geobench_root, small_partition):
        """Per-band values are raw uint16-ish DN, not normalized to [-5, 5] or [0, 1]."""
        bench = get_bench_dataset_class("m-eurosat")()
        dataset = bench.get_dataset(
            "train",
            partition=small_partition,
            bands=tuple(bench.rgb_bands),
        )

        sample = dataset[0]
        img = sample["image"]
        assert img.dtype.is_floating_point
        # Raw Sentinel-2 reflectance DN values are large.
        assert img.max() > 100.0, (
            f"Expected raw S2 magnitudes (max > 100), got max={img.max().item():.2f}; "
            "the dataset may still be normalizing internally."
        )


@pytest.mark.slow
class TestDataLoader:
    """Test integration with PyTorch DataLoader."""

    def test_dataloader_batching(self, geobench_root, small_partition):
        """Default collate stacks images/labels and keeps sample_ids as a list."""
        bench = get_bench_dataset_class("m-eurosat")()
        dataset = bench.get_dataset(
            "train",
            partition=small_partition,
            bands=tuple(bench.rgb_bands),
        )

        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(dataloader))

        assert batch["image"].shape[0] == 4
        assert batch["image"].shape[1] == 3
        assert batch["label"].shape[0] == 4
        assert isinstance(batch["sample_id"], list)
        assert len(batch["sample_id"]) == 4


@pytest.mark.slow
class TestBandSelection:
    """Test different band selections."""

    def test_all_bands(self, geobench_root, small_partition):
        """Loading all available bands through the wrapper."""
        bench = get_bench_dataset_class("m-eurosat")()
        dataset = bench.get_dataset("train", partition=small_partition, bands=None)
        assert dataset[0]["image"].shape[0] > 3


@pytest.mark.slow
class TestPartitions:
    """Test different partition sizes."""

    def test_partition_size_ordering(self, geobench_root):
        """Larger partitions have more samples."""
        bench = get_bench_dataset_class("m-eurosat")()
        partitions = ["0.01x_train", "0.02x_train", "0.05x_train", "0.10x_train"]
        sizes = [
            len(bench.get_dataset("train", partition=p, bands=tuple(bench.rgb_bands)))
            for p in partitions
        ]
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1], (
                f"Partition {partitions[i]} has {sizes[i]} samples but "
                f"{partitions[i + 1]} has {sizes[i + 1]} (expected more)"
            )


@pytest.mark.slow
class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_invalid_dataset_name(self, geobench_root):
        """Invalid dataset name raises FileNotFoundError at reader initialization."""
        with pytest.raises(FileNotFoundError):
            GeoBenchv1Sharded(
                root=geobench_root,
                dataset_name="m-nonexistent",
                split="train",
                partition="default",
                bands=EUROSAT_RGB_SOURCE_BANDS,
            )

    def test_invalid_partition(self, geobench_root):
        """Invalid partition raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            GeoBenchv1Sharded(
                root=geobench_root,
                dataset_name="m-eurosat",
                split="train",
                partition="nonexistent_partition",
                bands=EUROSAT_RGB_SOURCE_BANDS,
            )

    def test_invalid_split(self, geobench_root):
        """Invalid split raises ValueError."""
        with pytest.raises(ValueError, match="Split.*not found"):
            GeoBenchv1Sharded(
                root=geobench_root,
                dataset_name="m-eurosat",
                split="invalid_split",
                partition="default",
                bands=EUROSAT_RGB_SOURCE_BANDS,
            )

    def test_invalid_band_name_via_wrapper(self, geobench_root, small_partition):
        """Wrapper rejects unknown short band names eagerly with ValueError."""
        bench = get_bench_dataset_class("m-eurosat")()
        with pytest.raises(ValueError, match="unknown band"):
            bench.get_dataset("train", partition=small_partition, bands=("nonexistent_band",))
