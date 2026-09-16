"""Channel-selection tests for RESISC45, whose upstream loader has no ``bands`` argument.

The upstream class is mocked, so no dataset download is needed.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from torchgeo_bench.datasets import DatasetSpec, get_dataset_spec, load_split


class _FakeRESISC45:
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
def patched(monkeypatch: pytest.MonkeyPatch) -> DatasetSpec:
    monkeypatch.setattr("torchgeo.datasets.RESISC45", _FakeRESISC45)
    return get_dataset_spec("resisc45")


class TestMetadata:
    def test_declared_shape_matches_bands(self) -> None:
        bench = get_dataset_spec("resisc45")
        assert bench.num_channels == 3
        assert [b.name for b in bench.bands] == ["red", "green", "blue"]
        assert bench.rgb_indices == (0, 1, 2)

    def test_split_sizes_sum_to_the_published_total(self) -> None:
        # 45 classes x 700 images; the 60/20/20 split is torchgeo's.
        assert sum(get_dataset_spec("resisc45").split_sizes.values()) == 31500

    def test_statistics_are_uint8_scale(self) -> None:
        """Stats must stay in raw sensor units so ``model_native`` detects uint8."""
        from torchgeo_bench.models._input_units import InputUnit, detect_input_unit

        assert detect_input_unit(get_dataset_spec("resisc45").bands) is InputUnit.UINT8

    def test_uses_aerial_sensor_tag(self) -> None:
        """RESISC45 is tagged ``aerial`` so sensor-routed models accept it."""
        assert {band.sensor for band in get_dataset_spec("resisc45").bands} == {"aerial"}

    def test_data_root_is_fixed(self) -> None:
        # Compare Paths, not strings: str() is backslash-separated on Windows.
        assert Path(get_dataset_spec("resisc45").source.root) == Path("data/resisc45")


class TestBandSelection:
    def test_selection_uses_source_not_custom_metadata_order(self, patched: DatasetSpec) -> None:
        spec = replace(patched, bands=tuple(reversed(patched.bands)))
        loaded = load_split(spec, "train", bands="all")
        assert loaded.bands == tuple(reversed(patched.bands))
        torch.testing.assert_close(
            loaded.dataset[0]["image"][:, 0, 0], torch.tensor([2.0, 1.0, 0.0])
        )

    @pytest.mark.parametrize("bands", [None, ("red", "green", "blue")])
    def test_identity_selection_avoids_copying(
        self, patched: DatasetSpec, bands: tuple[str, ...] | None
    ) -> None:
        """Selecting all bands should avoid a per-sample channel copy."""
        ds = load_split("resisc45", "train", bands=bands).dataset
        assert ds.kwargs["transforms"].indices == (0, 1, 2)
        torch.testing.assert_close(ds[0]["image"][:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))

    @pytest.mark.parametrize(
        ("bands", "expected"),
        [(("blue", "red"), [2.0, 0.0]), (("blue", "green", "red"), [2.0, 1.0, 0.0])],
    )
    def test_selection_preserves_requested_order(
        self, patched: DatasetSpec, bands: tuple[str, ...], expected: list[float]
    ) -> None:
        loaded = load_split("resisc45", "train", bands=bands)
        ds = loaded.dataset
        assert [spec.name for spec in loaded.bands] == list(bands)
        image = ds[0]["image"]
        assert image.shape == (len(expected), 8, 8)
        torch.testing.assert_close(image[:, 0, 0], torch.tensor(expected))

    def test_unknown_band_is_rejected(self, patched: DatasetSpec) -> None:
        with pytest.raises(ValueError, match="unknown band 'nir'"):
            load_split("resisc45", "train", bands=("nir",))

    def test_unknown_split_is_rejected(self, patched: DatasetSpec) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            load_split("resisc45", "invalid")

    def test_split_is_forwarded_and_partition_rejected(self, patched: DatasetSpec) -> None:
        with pytest.raises(ValueError, match="does not support custom partitions"):
            load_split("resisc45", "test", partition="0.01x_train")
        ds = load_split("resisc45", "test", bands="all").dataset
        assert ds.kwargs["split"] == "test"
        assert ds.kwargs["download"] is False
        assert "partition" not in ds.kwargs


class TestTransformComposition:
    @pytest.mark.parametrize("bands", [("red",), ("blue", "green", "red")])
    def test_selection_and_raw_conversion_precede_resizing(
        self,
        patched: DatasetSpec,
        bands: tuple[str, ...],
    ) -> None:
        ds = load_split(patched, "train", bands=bands, image_size=4).dataset
        sample = ds[0]
        expected = torch.tensor([patched.bands.index(b) for b in patched.select_band_specs(bands)])
        torch.testing.assert_close(
            sample["image"], expected.float()[:, None, None].expand(-1, 4, 4)
        )
