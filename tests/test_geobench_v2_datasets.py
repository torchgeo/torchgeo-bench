"""Tests for the high-level get_datasets API for GeoBench V2 datasets."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets


class MockV2Dataset:
    """Stand-in for ``geobench_v2.datasets.GeoBench<X>`` upstream classes."""

    def __init__(self, root, split, transforms=None, band_order=None, **kwargs):
        del kwargs
        self.root = root
        self.split = split
        self.transforms = transforms
        self.band_order = band_order

        if isinstance(band_order, dict):
            self.c = sum(len(v) for v in band_order.values())
        elif band_order is not None:
            self.c = len(band_order)
        else:
            self.c = 3
        self.h, self.w = 32, 32

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        img = torch.randn(self.c, self.h, self.w)
        sample = {"image": img}
        if self.transforms:
            sample = self.transforms(sample)
        # Return both label and mask so segmentation+classification tests both pass.
        sample.setdefault("label", torch.tensor(1))
        sample.setdefault("mask", torch.randint(0, 2, (self.h, self.w)))
        return sample


@pytest.fixture
def mock_v2_env():
    """Patch the V2 dataset classes at their upstream module location."""
    with (
        patch(
            "geobench_v2.datasets.GeoBenchBENV2",
            MagicMock(side_effect=MockV2Dataset),
        ),
        patch(
            "geobench_v2.datasets.GeoBenchBurnScars",
            MagicMock(side_effect=MockV2Dataset),
        ),
        patch(
            "geobench_v2.datasets.GeoBenchSo2Sat",
            MagicMock(side_effect=MockV2Dataset),
        ),
    ):
        yield


class TestV2Loading:
    def test_benv2_classification(self, mock_v2_env):
        del mock_v2_env
        ds, train_dl, val_dl, test_dl = get_datasets(
            dataset_name="benv2",
            return_val=True,
            batch_size=4,
            num_workers=0,
        )

        assert isinstance(train_dl, DataLoader)
        assert len(ds) == 10
        assert get_bench_dataset_class("benv2").task == "classification"

        batch = next(iter(train_dl))
        assert batch["image"].shape == (4, 3, 32, 32)
        assert "label" in batch

    def test_burn_scars_segmentation(self, mock_v2_env):
        del mock_v2_env
        ds, train_dl, test_dl = get_datasets(
            dataset_name="burn_scars",
            batch_size=2,
            return_val=False,
            num_workers=0,
        )

        assert get_bench_dataset_class("burn_scars").task == "segmentation"
        batch = next(iter(train_dl))
        assert "mask" in batch
        assert batch["image"].shape[0] == 2

    def test_partition_warning(self, mock_v2_env):
        del mock_v2_env
        with pytest.warns(UserWarning, match="does not support custom partitions"):
            get_datasets(
                dataset_name="benv2",
                partition_name="0.10x_train",
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
        )

        # The resize transform is forwarded as ``transforms`` to the upstream
        # mock through ``GeoBenchv2._inner``.
        assert ds._inner.transforms is not None

        dl = DataLoader(ds, batch_size=1)
        batch = next(iter(dl))
        assert batch["image"].shape[-1] == target

    def test_bad_dataset_name(self):
        with pytest.raises(KeyError, match="Unknown dataset 'phantom_dataset'"):
            get_datasets(dataset_name="phantom_dataset")

    def test_no_double_root_join(self, mock_v2_env):
        """``GeoBenchv2`` must combine collection-root + dataset-name once."""
        with patch(
            "geobench_v2.datasets.GeoBenchBENV2",
            MagicMock(side_effect=MockV2Dataset),
        ) as mocked:
            del mock_v2_env
            get_datasets(
                dataset_name="benv2",
                batch_size=2,
                num_workers=0,
            )
            assert mocked.call_count == 3  # train, val, test
            for call in mocked.call_args_list:
                kwargs = call.kwargs
                assert Path(kwargs["root"]) == Path("data/geobenchv2/benv2"), kwargs

    def test_band_order_shape_dict(self, mock_v2_env):
        """Multi-modality V2 wrappers must hand a dict ``band_order`` upstream."""
        with patch(
            "geobench_v2.datasets.GeoBenchBENV2",
            MagicMock(side_effect=MockV2Dataset),
        ) as mocked:
            del mock_v2_env
            get_datasets(
                dataset_name="benv2",
                bands="rgb",
                batch_size=2,
                num_workers=0,
            )
            for call in mocked.call_args_list:
                bo = call.kwargs["band_order"]
                assert isinstance(bo, dict), bo
                assert bo == {"s2": ["B04", "B03", "B02"]}, bo

    def test_band_order_shape_flat(self, mock_v2_env):
        """Single-modality V2 wrappers must hand a flat list ``band_order`` upstream."""
        with patch(
            "geobench_v2.datasets.GeoBenchBurnScars",
            MagicMock(side_effect=MockV2Dataset),
        ) as mocked:
            del mock_v2_env
            get_datasets(
                dataset_name="burn_scars",
                bands="rgb",
                batch_size=2,
                num_workers=0,
            )
            for call in mocked.call_args_list:
                bo = call.kwargs["band_order"]
                assert isinstance(bo, list), bo
                assert bo == ["B04", "B03", "B02"], bo


class MockKuroSiwo:
    """Stand-in for ``geobench_v2.datasets.GeoBenchKuroSiwo``.

    Mirrors the real upstream loader's *unstacked* output shape: a per-modality
    dict containing ``image_pre_1`` / ``image_pre_2`` / ``image_post`` for SAR
    (gated by ``time_step``) and ``image_dem`` for DEM, plus ``mask`` and
    ``invalid_data``. Each tensor is 3-D ``(C, H, W)``.

    Channel counts come from ``band_order`` (a ``dict[modality, list[str]]``),
    which is what the wrapper's ``band_order_strategy = "by_sensor"`` produces.
    """

    def __init__(
        self,
        root,
        split,
        *,
        band_order=None,
        time_step=("pre_1", "pre_2", "post"),
        transforms=None,
        return_stacked_image=False,
        **kwargs,
    ):
        del kwargs
        self.root = root
        self.split = split
        self.band_order = band_order or {}
        self.time_step = list(time_step)
        self.transforms = transforms
        self.return_stacked_image = return_stacked_image
        self.h, self.w = 16, 16

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        del idx
        sample: dict[str, torch.Tensor] = {
            "invalid_data": torch.ones(1, self.h, self.w, dtype=torch.long),
            "mask": torch.zeros(self.h, self.w, dtype=torch.long),
        }
        if "sar" in self.band_order:
            sar_c = len(self.band_order["sar"])
            for ts in self.time_step:
                sample[f"image_{ts}"] = torch.ones(sar_c, self.h, self.w) * float(
                    {"pre_1": 1.0, "pre_2": 2.0, "post": 3.0}[ts]
                )
        if "dem" in self.band_order:
            sample["image_dem"] = torch.full((len(self.band_order["dem"]), self.h, self.w), 99.0)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class TestKuroSiwoCanonicalization:
    """Verify the kuro_siwo wrapper folds per-modality keys into a 3-D image."""

    @pytest.fixture
    def mocked_kuro_siwo(self):
        with patch(
            "geobench_v2.datasets.GeoBenchKuroSiwo",
            MagicMock(side_effect=MockKuroSiwo),
        ) as mocked:
            yield mocked

    @pytest.mark.parametrize(
        "bands,expected_channels",
        [
            (("vv", "vh"), 2),
            (("vv",), 1),
            (("dem",), 1),
            (("vv", "dem"), 2),
            (("vv", "vh", "dem"), 3),
            (None, 3),  # all bands
        ],
    )
    def test_image_is_3d_with_correct_channel_count(
        self, mocked_kuro_siwo, bands, expected_channels
    ):
        """For every band selection the canonical image must be ``(C, H, W)``."""
        bench = get_bench_dataset_class("kuro_siwo")()
        ds = bench.get_dataset("train", bands=bands)
        sample = ds[0]
        assert "image" in sample
        img = sample["image"]
        assert img.dim() == 3, f"expected 3-D image, got shape {tuple(img.shape)} for bands={bands}"
        assert img.shape[0] == expected_channels, (
            f"expected {expected_channels} channels, got {img.shape[0]} for bands={bands}"
        )

        for stale in ("image_pre_1", "image_pre_2", "image_post", "image_dem"):
            assert stale not in sample, f"per-modality key {stale!r} should be folded into 'image'"

        assert mocked_kuro_siwo.called

    def test_uses_post_event_sar_only(self, mocked_kuro_siwo):
        """The wrapper must request ``time_step=['post']`` from upstream."""
        bench = get_bench_dataset_class("kuro_siwo")()
        bench.get_dataset("train", bands=("vv", "vh"))
        for call in mocked_kuro_siwo.call_args_list:
            assert call.kwargs["time_step"] == ["post"], call.kwargs

    def test_does_not_request_stacked_image(self, mocked_kuro_siwo):
        """The wrapper must NOT request upstream's broken stacking path."""
        bench = get_bench_dataset_class("kuro_siwo")()
        bench.get_dataset("train", bands=None)
        for call in mocked_kuro_siwo.call_args_list:
            assert call.kwargs.get("return_stacked_image", False) is False, call.kwargs

    def test_dem_concatenated_after_sar(self, mocked_kuro_siwo):
        """When both SAR and DEM are requested, DEM lives in the trailing channels."""
        del mocked_kuro_siwo  # only used to install the mock
        bench = get_bench_dataset_class("kuro_siwo")()
        ds = bench.get_dataset("train", bands=("vv", "vh", "dem"))
        img = ds[0]["image"]
        # MockKuroSiwo paints SAR-post with 3.0 and DEM with 99.0
        assert torch.allclose(img[:2], torch.full_like(img[:2], 3.0))
        assert torch.allclose(img[2:], torch.full_like(img[2:], 99.0))


@pytest.mark.slow
class TestKuroSiwoLive:
    """Smoke tests against real Kuro Siwo data (skipped if the dataset is missing)."""

    @pytest.mark.parametrize(
        "bands,expected_channels",
        [
            (("vv", "vh"), 2),
            (None, 3),  # all bands: vv, vh, dem
            (("vv", "dem"), 2),
        ],
    )
    def test_real_sample_is_3d(self, geobench_v2_root, bands, expected_channels):
        """Loading real kuro_siwo data must yield a 3-D image with the expected channels."""
        del geobench_v2_root
        bench = get_bench_dataset_class("kuro_siwo")()
        ds = bench.get_dataset("train", bands=bands)
        sample = ds[0]
        img = sample["image"]
        assert img.dim() == 3, f"expected 3-D, got shape {tuple(img.shape)}"
        assert img.shape[0] == expected_channels, (
            f"expected {expected_channels} channels, got {img.shape[0]}"
        )
