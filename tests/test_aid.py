"""Tests for the AID loader; the upstream torchgeo class is mocked."""

from pathlib import Path

import pytest

from torchgeo_bench.datasets.aid import AID


class _FakeNonGeoClassificationDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def patched(monkeypatch):
    import torchgeo_bench.datasets.aid as mod

    monkeypatch.setattr(mod, "NonGeoClassificationDataset", _FakeNonGeoClassificationDataset)
    return AID.__new__(AID)


class TestMetadata:
    def test_declared_shape_matches_bands(self):
        bench = AID()
        assert bench.num_channels == 3
        assert [b.name for b in bench.bands] == ["red", "green", "blue"]

    def test_split_sizes_sum_to_the_published_total(self):
        assert sum(AID.split_sizes.values()) == 10000

    def test_data_root_is_fixed(self):
        assert AID.data_root() == Path("data/aid")


class TestGetDataset:
    def test_unknown_split_is_rejected(self, patched):
        with pytest.raises(ValueError, match="Unknown split"):
            patched.get_dataset("invalid")

    def test_unknown_band_is_rejected(self, patched):
        with pytest.raises(ValueError, match="unknown band 'nir'"):
            patched.get_dataset("train", bands=("nir",))

    def test_missing_split_file_raises(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="download aid"):
            patched.get_dataset("train")

    def test_root_and_is_valid_file_from_split_list(self, patched, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_root = tmp_path / "data" / "aid"
        data_root.mkdir(parents=True)
        (data_root / "aid-train.txt").write_text("airport_1.jpg\nbeach_2.jpg\n")

        ds = patched.get_dataset("train")

        assert ds.kwargs["root"] == str(Path("data/aid/AID"))
        is_valid_file = ds.kwargs["is_valid_file"]
        assert is_valid_file("/some/path/airport_1.jpg") is True
        assert is_valid_file("/some/path/other.jpg") is False
