"""Tests for the geobenchV2 dataset in datasets.py module and get_datasets function."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_datasets


class MockV2Dataset:
    def __init__(self, root, split, transforms=None, data_normalizer=None, band_order=None):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.data_normalizer = data_normalizer
        self.band_order = band_order

        # Determine channels based on bands
        self.c = len(band_order) if band_order else 3
        self.h, self.w = 32, 32

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        img = torch.randn(self.c, self.h, self.w)
        sample = {"image": img}

        if self.transforms:
            sample = self.transforms(sample)

        if getattr(self, "task_type", None) == "segmentation":
            sample["mask"] = torch.randint(0, 2, (self.h, self.w))
        else:
            sample["label"] = torch.tensor(1)

        return sample


@pytest.fixture
def mock_v2_env():
    with patch("torchgeo_bench.datasets.gb_v2") as mock_pkg:
        # Side_effect ensures we get fresh instances
        mock_pkg.GeoBenchBENV2 = MagicMock(side_effect=MockV2Dataset)
        mock_pkg.GeoBenchBiomassters = MagicMock(side_effect=MockV2Dataset)
        mock_pkg.GeoBenchSo2Sat = MagicMock(side_effect=MockV2Dataset)

        yield mock_pkg


class TestV2Loading:
    def test_benv2_classification(self, mock_v2_env):
        del mock_v2_env
        ds, train_dl, val_dl, test_dl = get_datasets(
            dataset_name="benv2",
            return_val=True,
            batch_size=4,
            num_workers=0,
            geobench_v2_root="/tmp/dummy",
        )

        assert isinstance(train_dl, DataLoader)
        assert len(ds) == 10
        assert ds.task_type == "classification"

        batch = next(iter(train_dl))
        assert batch["image"].shape == (4, 3, 32, 32)  # B, C, H, W
        assert "label" in batch

    @pytest.mark.skip(reason="Biomassters is pixelwise regression, not yet supported")
    def test_biomassters_segmentation(self, mock_v2_env):
        del mock_v2_env
        ds, train_dl, test_dl = get_datasets(
            dataset_name="biomassters",
            batch_size=2,
            return_val=False,
            num_workers=0,
            geobench_v2_root="/tmp/dummy",
        )

        assert ds.task_type == "segmentation"

        batch = next(iter(train_dl))
        assert "mask" in batch
        assert batch["image"].shape[0] == 2

    def test_partition_warning(self, mock_v2_env):
        del mock_v2_env
        with pytest.warns(UserWarning, match="Partitions are not supported in GeoBench V2"):
            get_datasets(
                dataset_name="benv2",
                partition_name="0.10x_train",
                geobench_v2_root="/tmp/dummy",
                num_workers=0,
            )

    def test_resize_transform(self, mock_v2_env):
        del mock_v2_env
        target = 64
        ds, _, _ = get_datasets(
            dataset_name="benv2",
            image_size=target,
            batch_size=4,
            num_workers=0,
            geobench_v2_root="/tmp/dummy",
        )

        assert ds.transforms is not None

        dl = DataLoader(ds, batch_size=1)
        batch = next(iter(dl))
        assert batch["image"].shape[-1] == target

    def test_bad_dataset_name(self, mock_v2_env):
        mock_v2_env.GeoBenchCaFFe = None

        with pytest.raises(ValueError, match="Could not find V2 dataset class"):
            get_datasets(dataset_name="caffe", geobench_v2_root="/tmp")
