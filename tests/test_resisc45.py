"""Unit tests for the RESISC45 wrapper.

RESISC45 is the reference example for wrapping a torchgeo dataset whose
loader has no ``bands`` argument, so these tests focus on the part the
wrapper owns that :class:`EuroSAT` does not: channel selection.  They run
without the dataset on disk by monkeypatching the upstream class.
"""

from pathlib import Path

import pytest
import torch

from torchgeo_bench.datasets.resisc45 import RESISC45, _make_band_select


class _FakeRESISC45:
    """Stand-in for ``torchgeo.datasets.RESISC45``: 3-channel, values 0/1/2."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.transforms = kwargs.get("transforms")

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict:
        # Channel c is filled with the constant c, so a selected sample's
        # values name the source channel it came from.
        image = torch.arange(3, dtype=torch.float32).view(3, 1, 1).expand(3, 8, 8).clone()
        sample = {"image": image, "label": torch.tensor(index)}
        return self.transforms(sample) if self.transforms is not None else sample


@pytest.fixture
def patched(monkeypatch):
    import torchgeo_bench.datasets.resisc45 as mod

    monkeypatch.setattr(mod, "TGRESISC45", _FakeRESISC45)
    return RESISC45.__new__(RESISC45)


class TestMetadata:
    def test_declared_shape_matches_bands(self):
        bench = RESISC45()
        assert bench.num_channels == 3
        assert [b.name for b in bench.bands] == ["red", "green", "blue"]
        assert bench.rgb_indices == [0, 1, 2]

    def test_split_sizes_sum_to_the_published_total(self):
        # 45 classes x 700 images; the 60/20/20 split is torchgeo's.
        assert sum(RESISC45.split_sizes.values()) == 31500

    def test_statistics_are_uint8_scale(self):
        """Stats must stay in raw sensor units so ``model_native`` detects uint8."""
        from torchgeo_bench.models._input_units import InputUnit, detect_input_unit

        assert detect_input_unit(RESISC45().bands) is InputUnit.UINT8

    def test_uses_aerial_sensor_tag(self):
        """RESISC45 is tagged ``aerial`` so sensor-routed models accept it."""
        assert {band.sensor for band in RESISC45.bands} == {"aerial"}

    def test_data_root_is_fixed(self):
        # Compare Paths, not strings: str() is backslash-separated on Windows.
        assert RESISC45.data_root() == Path("data/resisc45")


class TestBandSelection:
    def test_all_bands_adds_no_transform(self, patched):
        """The common path must not pay for a per-sample index_select."""
        ds = patched.get_dataset("train", bands=None)
        assert ds.kwargs["transforms"] is None

    def test_rgb_in_declared_order_adds_no_transform(self, patched):
        ds = patched.get_dataset("train", bands=("red", "green", "blue"))
        assert ds.kwargs["transforms"] is None

    def test_subset_selects_the_right_channels(self, patched):
        ds = patched.get_dataset("train", bands=("blue", "red"))
        image = ds[0]["image"]
        assert image.shape == (2, 8, 8)
        # blue is source channel 2, red is source channel 0.
        assert torch.equal(image[:, 0, 0], torch.tensor([2.0, 0.0]))

    def test_reordering_is_honoured(self, patched):
        ds = patched.get_dataset("train", bands=("blue", "green", "red"))
        assert torch.equal(ds[0]["image"][:, 0, 0], torch.tensor([2.0, 1.0, 0.0]))

    def test_unknown_band_is_rejected(self, patched):
        with pytest.raises(ValueError, match="unknown band 'nir'"):
            patched.get_dataset("train", bands=("nir",))

    def test_split_is_forwarded_and_partition_ignored(self, patched):
        ds = patched.get_dataset("test", partition="0.01x_train", bands=None)
        assert ds.kwargs["split"] == "test"
        assert "partition" not in ds.kwargs


class TestTransformComposition:
    def test_selection_runs_before_the_caller_transform(self, patched):
        """The resize must see only the channels that survive selection."""
        seen: list[tuple[int, ...]] = []

        def _resize(sample: dict) -> dict:
            seen.append(tuple(sample["image"].shape))
            return sample

        ds = patched.get_dataset("train", bands=("red",), transform=_resize)
        ds[0]
        assert seen == [(1, 8, 8)]

    def test_caller_transform_survives_when_selection_is_identity(self, patched):
        calls: list[int] = []

        def _mark(sample: dict) -> dict:
            calls.append(1)
            return sample

        ds = patched.get_dataset("train", bands=None, transform=_mark)
        assert ds.kwargs["transforms"] is _mark
        ds[0]
        assert calls == [1]

    def test_band_select_identity_is_none(self):
        assert _make_band_select([0, 1, 2], 3) is None
        assert _make_band_select([0, 2], 3) is not None
