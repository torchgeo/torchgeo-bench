"""Unit tests for dataset classes that don't require real data on disk."""

import pickle
from pathlib import Path

import pytest
import torch

from torchgeo_bench.datasets._metadata import unpickle_metadata
from torchgeo_bench.datasets.eurosat import EuroSAT, EuroSATSpatial
from torchgeo_bench.datasets.fotw import FieldsOfTheWorld as FOTW
from torchgeo_bench.datasets.geobench_v2 import _V2Dataset
from torchgeo_bench.datasets.spacenet2 import SpaceNet2
from torchgeo_bench.datasets.spacenet7 import SpaceNet7


def test_unpickle_metadata_accepts_bytes_and_repr() -> None:
    metadata = {"label": 3, "bands_order": ["red", "green", "blue"]}
    payload = pickle.dumps(metadata)

    assert unpickle_metadata(payload) == metadata
    assert unpickle_metadata(repr(payload)) == metadata


def test_unpickle_metadata_rejects_python_expressions() -> None:
    with pytest.raises((SyntaxError, ValueError)):
        unpickle_metadata("__import__('os').system('echo unsafe')")


def test_v2_data_root_is_fixed() -> None:
    assert _V2Dataset.data_root() == Path("data/geobenchv2")


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


class TestSpaceNet2Canonicalize:
    def test_mask_offset_reversed(self):
        """Upstream ``{1: no-building, 2: building}`` maps to native ``{0, 1}``."""
        ds = SpaceNet2.__new__(SpaceNet2)
        mask = torch.tensor([[1, 2], [2, 1]])
        result = ds.canonicalize_sample({"image": torch.zeros(3, 2, 2), "mask": mask})
        assert torch.equal(result["mask"], torch.tensor([[0, 1], [1, 0]]))
        assert int(result["mask"].max()) < SpaceNet2.num_classes

    def test_reserved_zero_folds_into_no_building(self):
        """A stray reserved class 0 clamps to 0 (no-building), never wraps to -1."""
        ds = SpaceNet2.__new__(SpaceNet2)
        mask = torch.tensor([[0, 1, 2]])
        result = ds.canonicalize_sample({"image": torch.zeros(3, 1, 3), "mask": mask})
        assert torch.equal(result["mask"], torch.tensor([[0, 0, 1]]))

    def test_missing_mask_is_noop(self):
        """No mask key (e.g. inference) leaves the sample untouched."""
        ds = SpaceNet2.__new__(SpaceNet2)
        sample = {"image": torch.zeros(3, 4, 4)}
        assert ds.canonicalize_sample(sample) == sample


class TestSpaceNet7Canonicalize:
    def test_mask_offset_reversed(self):
        """Upstream ``{1: no-building, 2: building}`` maps to native ``{0, 1}``."""
        ds = SpaceNet7.__new__(SpaceNet7)
        mask = torch.tensor([[1, 2], [2, 1]])
        result = ds.canonicalize_sample({"image": torch.zeros(3, 2, 2), "mask": mask})
        assert torch.equal(result["mask"], torch.tensor([[0, 1], [1, 0]]))
        assert int(result["mask"].max()) < SpaceNet7.num_classes

    def test_reserved_zero_folds_into_no_building(self):
        """A stray reserved class 0 clamps to 0 (no-building), never wraps to -1."""
        ds = SpaceNet7.__new__(SpaceNet7)
        mask = torch.tensor([[0, 1, 2]])
        result = ds.canonicalize_sample({"image": torch.zeros(3, 1, 3), "mask": mask})
        assert torch.equal(result["mask"], torch.tensor([[0, 0, 1]]))

    def test_missing_mask_is_noop(self):
        """No mask key (e.g. inference) leaves the sample untouched."""
        ds = SpaceNet7.__new__(SpaceNet7)
        sample = {"image": torch.zeros(3, 4, 4)}
        assert ds.canonicalize_sample(sample) == sample


# ---------------------------------------------------------------------------
# EuroSAT / EuroSATSpatial metadata
# ---------------------------------------------------------------------------


class TestEuroSATMeta:
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
