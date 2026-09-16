"""Channel-selection tests for UCMerced, whose upstream loader has no ``bands`` argument.

The upstream class is mocked, so no dataset download is needed.
"""

import pickle
from pathlib import Path

import pytest
import torch

from torchgeo_bench.datasets import get_bench_dataset_class, list_datasets
from torchgeo_bench.datasets.loading import _ResizeTransform
from torchgeo_bench.datasets.ucmerced import UCMerced


class _FakeUCMerced:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.transforms = kwargs.get("transforms")

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        # Pixel values identify the source channel after selection.
        image = torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 8, 8).clone()
        sample = {"image": image, "label": torch.tensor(index)}
        return self.transforms(sample) if self.transforms is not None else sample


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch) -> UCMerced:
    import torchgeo_bench.datasets.ucmerced as mod

    monkeypatch.setattr(mod, "TGUCMerced", _FakeUCMerced)
    return UCMerced()


class TestMetadata:
    def test_registration(self) -> None:
        from torchgeo_bench.datasets import UCMerced as exported

        assert exported is UCMerced
        assert get_bench_dataset_class("ucmerced") is UCMerced
        assert "ucmerced" in list_datasets()

    def test_declared_shape_matches_bands(self) -> None:
        bench = UCMerced()
        assert bench.num_channels == 3
        assert [b.name for b in bench.bands] == ["red", "green", "blue"]
        assert bench.rgb_indices == [0, 1, 2]

    def test_classification_metadata(self) -> None:
        assert UCMerced.task == "classification"
        assert UCMerced.num_classes == 21
        assert UCMerced.multilabel is False
        assert UCMerced.supports_partitions is False

    def test_split_sizes_sum_to_the_published_total(self) -> None:
        # 21 classes x 100 images; the 60/20/20 split is torchgeo's.
        assert sum(UCMerced.split_sizes.values()) == 2100

    def test_statistics_are_uint8_scale(self) -> None:
        """Stats must stay in raw sensor units so ``model_native`` detects uint8."""
        from torchgeo_bench.models._input_units import InputUnit, detect_input_unit

        assert detect_input_unit(UCMerced().bands) is InputUnit.UINT8

    def test_uses_aerial_sensor_tag(self) -> None:
        """UCMerced is tagged ``aerial`` so sensor-routed models accept it."""
        assert {band.sensor for band in UCMerced.bands} == {"aerial"}

    def test_data_root_is_fixed(self) -> None:
        assert UCMerced.data_root() == Path("data/ucmerced")


class TestBandSelection:
    @pytest.mark.parametrize("bands", [None, ("red", "green", "blue")])
    def test_identity_selection_avoids_copying(
        self, patched: UCMerced, bands: tuple[str, ...] | None
    ) -> None:
        ds = patched.get_dataset("train", bands=bands)
        assert ds.kwargs["transforms"] is None
        torch.testing.assert_close(ds[0]["image"][:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))

    @pytest.mark.parametrize(
        ("bands", "expected"),
        [
            (("red",), [0.0]),
            (("blue", "red"), [2.0, 0.0]),
            (("blue", "green", "red"), [2.0, 1.0, 0.0]),
        ],
    )
    def test_selection_preserves_requested_order(
        self, patched: UCMerced, bands: tuple[str, ...], expected: list[float]
    ) -> None:
        ds = patched.get_dataset("train", bands=bands)
        sample = ds[2]
        assert sample["image"].shape == (len(expected), 8, 8)
        torch.testing.assert_close(sample["image"][:, 0, 0], torch.tensor(expected))
        assert sample["label"].item() == 2

    def test_unknown_band_is_rejected(self, patched: UCMerced) -> None:
        with pytest.raises(ValueError, match="unknown band 'nir'"):
            patched.get_dataset("train", bands=("nir",))

    def test_unknown_split_is_rejected(self, patched: UCMerced) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            patched.get_dataset("invalid")

    @pytest.mark.parametrize("split", ["train", "val", "test"])
    def test_split_is_forwarded_and_partition_ignored(self, patched: UCMerced, split: str) -> None:
        ds = patched.get_dataset(split, partition="0.01x_train", bands=None)
        assert ds.kwargs["split"] == split
        assert "partition" not in ds.kwargs


class TestTransformComposition:
    def test_selection_runs_before_the_caller_transform(self, patched: UCMerced) -> None:
        seen: list[tuple[int, ...]] = []

        def _resize(sample: dict) -> dict:
            seen.append(tuple(sample["image"].shape))
            return sample

        ds = patched.get_dataset("train", bands=("red",), transform=_resize)
        ds[0]
        assert seen == [(1, 8, 8)]

    @pytest.mark.parametrize("bands", [None, ("red", "green", "blue")])
    def test_caller_transform_survives_when_selection_is_identity(
        self, patched: UCMerced, bands: tuple[str, ...] | None
    ) -> None:
        calls: list[int] = []

        def _mark(sample: dict) -> dict:
            calls.append(1)
            return sample

        ds = patched.get_dataset("train", bands=bands, transform=_mark)
        assert ds.kwargs["transforms"] is _mark
        ds[0]
        assert calls == [1]

    def test_composed_transform_survives_pickling(self, patched: UCMerced) -> None:
        ds = patched.get_dataset(
            "train", bands=("blue", "red"), transform=_ResizeTransform(12, "bilinear")
        )
        ds.transforms = pickle.loads(pickle.dumps(ds.transforms))
        image = ds[0]["image"]
        assert image.shape == (2, 12, 12)
        torch.testing.assert_close(image[:, 0, 0], torch.tensor([2.0, 0.0]))
