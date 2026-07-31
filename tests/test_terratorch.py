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


def _bands(names: list[str], sensor: str = "s2") -> list[BandSpec]:
    return [
        BandSpec(
            sensor=sensor,
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
            self.last_forward_input: object | None = None

        def forward(self, x, **kwargs):
            self.last_forward_kwargs = kwargs
            self.last_forward_input = x
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


_S2L2A_FULL = [
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


def test_terramind_modality_shape(mock_registry):
    bands = _bands(_S2L2A_FULL)
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity")
    out = model.forward_patch_features(torch.rand(2, len(bands), 224, 224))
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_terramind_full_s2l2a_no_band_selection(mock_registry):
    bands = _bands(_S2L2A_FULL)
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["modalities"] == ["S2L2A"]
    assert "bands" not in build_kwargs
    x = torch.rand(1, 12, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert payload.shape == (1, 12, 224, 224)
    assert torch.equal(payload, x)


def test_terramind_incomplete_s2l2a_uses_band_selection(mock_registry):
    names = ["b02", "b03", "b04", "b05", "b06", "b07", "b08", "b8a", "b11", "b12"]
    model = TerraTorchTerraMindBench(bands=_bands(names), normalization="identity")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["bands"] == {
        "S2L2A": [
            "BLUE",
            "GREEN",
            "RED",
            "RED_EDGE_1",
            "RED_EDGE_2",
            "RED_EDGE_3",
            "NIR_BROAD",
            "NIR_NARROW",
            "SWIR_1",
            "SWIR_2",
        ]
    }
    x = torch.rand(2, 10, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert payload.shape == (2, 10, 224, 224)
    assert torch.equal(payload, x)


def test_terramind_native_rgb_modality_reorders_channels(mock_registry):
    bands = _bands(["blue", "green", "red"])
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity", modality="RGB")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["modalities"] == ["RGB"]
    assert "bands" not in build_kwargs
    x = torch.rand(1, 3, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["RGB"]
    assert torch.equal(payload[:, 0], x[:, 2])
    assert torch.equal(payload[:, 1], x[:, 1])
    assert torch.equal(payload[:, 2], x[:, 0])


def test_terramind_rgb_dataset_with_s2l2a_modality_selects_three_bands(mock_registry):
    bands = _bands(["red", "green", "blue"])
    model = TerraTorchTerraMindBench(bands=bands, normalization="identity")
    _, build_kwargs = mock_registry["build_calls"][0]
    assert build_kwargs["bands"] == {"S2L2A": ["BLUE", "GREEN", "RED"]}
    model.forward_patch_features(torch.rand(1, 3, 224, 224))
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert payload.shape[1] == 3


def test_terramind_duplicate_canonical_names_prefer_s2(mock_registry):
    aerial = _bands(["red", "green", "blue", "nir"], sensor="aerial")
    s2 = _bands(["b02", "b03", "b04", "b08"])
    model = TerraTorchTerraMindBench(bands=aerial + s2, normalization="identity")
    x = torch.arange(8.0).view(1, 8, 1, 1).expand(1, 8, 224, 224)
    model.forward_patch_features(x)
    payload = mock_registry["instances"][-1].last_forward_input["S2L2A"]
    assert torch.equal(payload, x[:, [4, 5, 6, 7]])


def test_terramind_model_native_rgb_applies_pretraining_stats(mock_registry):
    bands = [
        BandSpec(sensor="aerial", name=n, source_name=n, mean=120.0, std=50.0, min=0.0, max=255.0)
        for n in ["red", "green", "blue"]
    ]
    model = TerraTorchTerraMindBench(bands=bands, normalization="model_native", modality="RGB")
    x = torch.full((1, 3, 4, 4), 100.0)
    out = model.normalize_inputs(x)
    # TerraMind v1 RGB pretraining stats: RED (87.271, 58.767), BLUE (66.667, 42.631)
    assert torch.allclose(out[:, 0], torch.full((1, 4, 4), (100.0 - 87.271) / 58.767), atol=1e-4)
    assert torch.allclose(out[:, 2], torch.full((1, 4, 4), (100.0 - 66.667) / 42.631), atol=1e-4)


def test_terramind_model_native_s2l2a_converts_unit_then_zscores(mock_registry):
    bands = _bands(["blue", "green", "red"])
    model = TerraTorchTerraMindBench(bands=bands, normalization="model_native")
    x = torch.full((1, 3, 4, 4), 0.15)
    out = model.normalize_inputs(x)
    # 0.15 reflectance -> 1500 DN; S2L2A BLUE pretraining stats (1503.317, 2141.107)
    assert torch.allclose(
        out[:, 0], torch.full((1, 4, 4), (1500.0 - 1503.317) / 2141.107), atol=1e-4
    )


def test_terramind_unsupported_modality_raises(mock_registry):
    with pytest.raises(ValueError, match="Unsupported TerraMind modality"):
        TerraTorchTerraMindBench(
            bands=_bands(["red", "green", "blue"]), normalization="identity", modality="DEM"
        )


def test_terramind_rgb_modality_without_rgb_bands_raises(mock_registry):
    sar = _bands(["vv", "vh"], sensor="sar")
    with pytest.raises(ValueError, match="none of the target bands"):
        TerraTorchTerraMindBench(bands=sar, normalization="identity", modality="RGB")


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
