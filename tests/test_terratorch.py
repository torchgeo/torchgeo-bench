"""Unit tests for TerraTorch model wrappers with mocked registry backbones."""

from importlib.util import find_spec

import pytest
import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models.terratorch_models import (
    TerraTorchClayBench,
    TerraTorchPrithviBench,
    TerraTorchTerraMindBench,
)

terratorch_available = find_spec("terratorch") is not None
requires_terratorch = pytest.mark.skipif(
    not terratorch_available,
    reason="terratorch not installed",
)
pytestmark = [requires_terratorch]


def _bands(names: list[str]) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=0.2,
            std=0.1,
            min=0.0,
            max=1.0,
        )
        for name in names
    ]


@pytest.fixture
def mock_registry(monkeypatch):
    state: dict[str, object] = {"build_calls": [], "instances": []}

    class _FakeBackbone(nn.Module):
        def __init__(self, name: str, build_kwargs: dict[str, object]) -> None:
            super().__init__()
            self._name = name
            self._build_kwargs = build_kwargs
            self.last_forward_kwargs: dict[str, object] | None = None

        def forward(self, x, **kwargs):
            self.last_forward_kwargs = kwargs
            if isinstance(x, dict):
                payload = next(iter(x.values()))
                batch = payload.shape[0]
            else:
                batch = x.shape[0]
            cls = torch.full((batch, 1, 8), 2.0)
            patches = torch.ones(batch, 4, 8)
            return torch.cat([cls, patches], dim=1)

    def _fake_build(name: str, **kwargs):
        instance = _FakeBackbone(name=name, build_kwargs=kwargs)
        state["build_calls"].append((name, kwargs))
        state["instances"].append(instance)
        return instance

    monkeypatch.setattr("torchgeo_bench.models.terratorch_models._build_backbone", _fake_build)
    return state


def test_prithvi_input_shape_accepted(mock_registry):
    bands = _bands(["blue", "green", "red", "nir_narrow", "swir1", "swir2"])
    model = TerraTorchPrithviBench(bands=bands, normalization="identity")
    out = model.forward_patch_features(torch.rand(2, len(bands), 224, 224))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()
    assert mock_registry["build_calls"][0][0] == "prithvi_eo_v2_300"


def test_clay_auxiliary_args_forwarded(mock_registry):
    bands = _bands(["blue", "green", "red", "nir", "swir1", "swir2"])
    model = TerraTorchClayBench(bands=bands, normalization="identity", gsd=20.0)
    out = model.forward_patch_features(torch.rand(2, len(bands), 256, 256))
    assert out.shape == (2, 8)
    instance = mock_registry["instances"][-1]
    kwargs = instance.last_forward_kwargs
    assert kwargs is not None
    assert "waves" in kwargs
    assert kwargs["waves"].shape == (6,)
    assert float(kwargs["gsd"]) == 20.0


def test_terramind_modality_shape(mock_registry):
    bands = _bands(
        [
            "coastal",
            "blue",
            "green",
            "red",
            "rededge1",
            "rededge2",
            "rededge3",
            "nir",
            "nir_narrow",
            "watervapor",
            "swir1",
            "swir2",
        ]
    )
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity")
    out = model.forward_patch_features(torch.rand(2, len(bands), 224, 224))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_pooling_mode_mean(mock_registry):
    bands = _bands(["blue", "green", "red", "nir_narrow", "swir1", "swir2"])
    model = TerraTorchPrithviBench(bands=bands, normalization="identity", pool="mean")
    out = model.forward_patch_features(torch.rand(2, len(bands), 224, 224))
    assert out.shape == (2, 8)


def test_pooling_mode_cls(mock_registry):
    bands = _bands(["blue", "green", "red", "nir_narrow", "swir1", "swir2"])
    model = TerraTorchPrithviBench(bands=bands, normalization="identity", pool="cls")
    out = model.forward_patch_features(torch.rand(2, len(bands), 224, 224))
    assert out.shape == (2, 8)


def test_invalid_pool_mode_raises(mock_registry):
    bands = _bands(["blue", "green", "red", "nir_narrow", "swir1", "swir2"])
    with pytest.raises(ValueError, match="pool"):
        TerraTorchPrithviBench(bands=bands, normalization="identity", pool="bogus")
