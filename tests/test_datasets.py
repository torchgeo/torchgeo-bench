"""Tests for the datasets.py module and get_datasets function."""

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets


class TestGetDatasetsFunction:
    """Test the get_datasets factory function."""

    def test_get_datasets_basic(self, geobench_root):
        """Test basic get_datasets call."""
        result = get_datasets(
            dataset_name="m-eurosat",
            partition_name="0.01x_train",
            batch_size=8,
            return_val=False,
        )

        assert len(result) == 3  # train_dataset, train_loader, test_loader
        train_dataset, train_loader, test_loader = result

        assert len(train_dataset) > 0
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    def test_get_datasets_with_val(self, geobench_root):
        """Test get_datasets with return_val=True."""
        result = get_datasets(
            dataset_name="m-eurosat",
            partition_name="0.01x_train",
            batch_size=8,
            return_val=True,
        )

        assert len(result) == 4  # train_dataset, train_loader, val_loader, test_loader
        train_dataset, train_loader, val_loader, test_loader = result

        assert len(train_dataset) > 0
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)

    @pytest.mark.parametrize(
        "dataset_name",
        [
            "m-eurosat",
            "m-forestnet",
            "m-so2sat",
            "m-pv4ger",
            "m-brick-kiln",
        ],
    )
    def test_all_datasets_with_small_partition(self, geobench_root, dataset_name):
        """Test that get_datasets works for all datasets with 0.01x partition."""
        result = get_datasets(
            dataset_name=dataset_name,
            partition_name="0.01x_train",
            batch_size=4,
            return_val=True,
        )

        train_dataset, train_loader, val_loader, test_loader = result  # type: ignore[misc]

        # Check datasets are not empty
        assert len(train_dataset) > 0, f"{dataset_name}: Empty train dataset"

        # Check we can get batches
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        test_batch = next(iter(test_loader))

        # Verify batch shapes
        assert train_batch["image"].shape[1] == 3, f"{dataset_name}: Expected 3 channels"
        assert val_batch["image"].shape[1] == 3, f"{dataset_name}: Expected 3 channels"
        assert test_batch["image"].shape[1] == 3, f"{dataset_name}: Expected 3 channels"

        # Verify labels are in valid range
        expected_classes = get_bench_dataset_class(dataset_name).num_classes
        assert train_batch["label"].min() >= 0
        assert train_batch["label"].max() < expected_classes

    def test_raw_emission(self, geobench_root):
        """Datasets always emit raw float32 (normalization moved to BenchModel)."""
        result = get_datasets(
            dataset_name="m-eurosat",
            partition_name="0.01x_train",
            batch_size=4,
            return_val=False,
        )

        train_dataset, train_loader, test_loader = result  # type: ignore[misc]
        batch = next(iter(train_loader))

        assert batch["image"].dtype == torch.float32
        # Raw S2 reflectance DN values are large (well above the [-5, 5] z-score range).
        assert batch["image"].max().item() > 100.0


class TestIntegrationWithBenchmark:
    """Test integration scenarios similar to actual benchmark usage."""

    def test_benchmark_workflow(self, geobench_root):
        """Test a complete workflow similar to the benchmark script."""
        # Load datasets
        train_dataset, train_loader, val_loader, test_loader = get_datasets(  # type: ignore[misc]
            dataset_name="m-eurosat",
            partition_name="0.01x_train",
            batch_size=4,
            return_val=True,
        )

        # Get channel count from first sample
        first_sample = train_dataset[0]
        num_channels = first_sample["image"].shape[0]
        assert num_channels == 3

        # Collect some features (simulate embedding)
        all_features = []
        all_labels = []

        for batch in train_loader:
            images = batch["image"]
            labels = batch["label"]

            # Simulate feature extraction (just flatten for test)
            features = images.mean(dim=(2, 3))  # (B, C)

            all_features.append(features)
            all_labels.append(labels)

            if len(all_features) >= 3:  # Just do a few batches
                break

        features_tensor = torch.cat(all_features, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        assert features_tensor.shape[0] == labels_tensor.shape[0]
        assert features_tensor.shape[1] == num_channels
