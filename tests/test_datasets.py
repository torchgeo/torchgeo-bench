"""Tests for the datasets.py module and get_datasets function."""

import pytest
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_datasets


class TestGetDatasetsFunction:
    """Test the get_datasets factory function."""

    @pytest.mark.slow
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
