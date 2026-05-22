"""Unit tests for dataset classes that don't require real data on disk."""

import torch

from torchgeo_bench.datasets.eurosat import EuroSAT, EuroSATSpatial
from torchgeo_bench.datasets.fotw import FieldsOfTheWorld as FOTW

# ---------------------------------------------------------------------------
# FOTW.canonicalize_sample
# ---------------------------------------------------------------------------


class TestFOTWCanonicalize:
    def test_image_already_present(self):
        """If 'image' key exists, sample is returned unchanged."""
        ds = FOTW.__new__(FOTW)
        sample = {"image": torch.zeros(3, 8, 8), "label": 0}
        result = ds.canonicalize_sample(sample)
        assert "image" in result

    def test_image_b_becomes_image(self):
        """If only 'image_b' is present it should become 'image'."""
        ds = FOTW.__new__(FOTW)
        img_b = torch.ones(3, 8, 8)
        sample = {"image_b": img_b, "image_a": torch.zeros(3, 8, 8), "label": 1}
        result = ds.canonicalize_sample(sample)
        assert "image" in result
        assert torch.equal(result["image"], img_b)
        assert "image_a" not in result
        assert "image_b" not in result

    def test_image_b_only_no_image_a(self):
        """image_a may be absent; should still work."""
        ds = FOTW.__new__(FOTW)
        img_b = torch.full((3, 4, 4), 5.0)
        sample = {"image_b": img_b, "label": 2}
        result = ds.canonicalize_sample(sample)
        assert torch.equal(result["image"], img_b)


# ---------------------------------------------------------------------------
# EuroSAT / EuroSATSpatial metadata
# ---------------------------------------------------------------------------


class TestEuroSATMeta:
    def test_name(self):
        assert EuroSAT.name == "eurosat"

    def test_split_sizes(self):
        s = EuroSAT.split_sizes
        assert s["train"] + s["val"] + s["test"] == 27000

    def test_data_root(self):
        assert EuroSAT.data_root().name == "eurosat"

    def test_num_classes(self):
        assert EuroSAT.num_classes == 10

    def test_band_specs_non_empty(self):
        assert len(EuroSAT.bands) > 0

    def test_get_dataset_mocked(self, monkeypatch):
        """get_dataset calls TGEuroSAT with correct band codes — test without disk."""

        import torchgeo_bench.datasets.eurosat as mod

        captured = {}

        class _FakeDS:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "TGEuroSAT", _FakeDS)
        ds_inst = EuroSAT.__new__(EuroSAT)
        ds_inst.get_dataset("train", bands=("red", "green", "blue"))
        assert "split" in captured
        assert captured["split"] == "train"
        assert isinstance(captured["bands"], tuple)


class TestEuroSATSpatialMeta:
    def test_name(self):
        assert EuroSATSpatial.name == "eurosat-spatial"

    def test_split_sizes(self):
        s = EuroSATSpatial.split_sizes
        assert s["train"] + s["val"] + s["test"] == 27000

    def test_data_root_shared(self):
        # Both classes share the same data root
        assert EuroSAT.data_root() == EuroSATSpatial.data_root()

    def test_get_dataset_mocked(self, monkeypatch):

        import torchgeo_bench.datasets.eurosat as mod

        captured = {}

        class _FakeDS:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(mod, "TGEuroSATSpatial", _FakeDS)
        ds_inst = EuroSATSpatial.__new__(EuroSATSpatial)
        ds_inst.get_dataset("test", bands=None)
        assert captured["split"] == "test"
