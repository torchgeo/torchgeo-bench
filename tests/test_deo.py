"""Offline tests for the DEO TorchGeo wrapper."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_module

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models import torchgeo_models as tg_models
from torchgeo_bench.models.torchgeo_models import TorchGeoDEOBench


def _band(sensor: str, name: str, maximum: float) -> BandSpec:
    return BandSpec(sensor, name, name, mean=maximum / 2, std=1, min=0, max=maximum)


class _FakeDEO(nn.Module):
    def __init__(self, *, incompatible: bool = False) -> None:
        super().__init__()
        self.incompatible = incompatible
        self.last_input: torch.Tensor | None = None

    def load_state_dict(self, state_dict, strict: bool = True):
        del state_dict, strict
        if self.incompatible:
            return SimpleNamespace(missing_keys=["feat_extr.conv_rgb.0.weight"], unexpected_keys=[])
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_input = images
        values = torch.arange(1024, device=images.device, dtype=images.dtype)
        return values.view(1, 1, 1, 1024).expand(images.shape[0], 2, 3, -1)


def _mock_loader(monkeypatch, *, incompatible: bool = False) -> _FakeDEO:
    backbone = _FakeDEO(incompatible=incompatible)

    class _Weights:
        url = tg_models._DEO_CHECKPOINT_URL

        @staticmethod
        def get_state_dict(*, progress: bool, weights_only: bool):
            assert progress is True
            assert weights_only is True
            return {"checkpoint": torch.tensor(1)}

    monkeypatch.setattr(tg_models, "DEO_Weights", SimpleNamespace(DEO_SWIN=_Weights()))
    monkeypatch.setattr(tg_models, "deo_base", lambda *, weights: backbone)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: "/tmp/nonexistent-deo-cache")
    return backbone


@pytest.mark.parametrize(
    ("maximum", "raw", "expected"),
    [
        (10000, [10000.0, 5000.0, -1.0], [1.0, 0.5, 0.0]),
        (255, [255.0, 127.5, -1.0], [1.0, 0.5, 0.0]),
        (1, [1.0, 0.5, -1.0], [1.0, 0.5, 0.0]),
    ],
)
def test_deo_rgb_native_conversion_and_normalization(monkeypatch, maximum, raw, expected) -> None:
    _mock_loader(monkeypatch)
    bands = [_band("rgb", name, maximum) for name in ("red", "green", "blue")]
    model = TorchGeoDEOBench(bands=bands, mode="rgb")
    images = torch.tensor(raw).view(1, 3, 1, 1)

    actual = model.normalize_inputs(images)
    mean = torch.tensor(tg_models._DEO_RGB_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(tg_models._DEO_RGB_STD).view(1, 3, 1, 1)
    assert torch.allclose(actual, (torch.tensor(expected).view(1, 3, 1, 1) - mean) / std)


def _s2_bands(maximum: float, *, include_aerial: bool = False) -> list[BandSpec]:
    bands = []
    if include_aerial:
        bands.extend(_band("aerial", name, 255) for name in ("red", "green", "blue"))
    names = ("b04", "b03", "b02", "b05", "b06", "b07", "b08", "b8a", "b11", "b12")
    bands.extend(_band("s2", name, maximum) for name in names)
    return bands


def test_deo_s2_prefers_s2_semantics_and_scales_rgb_per_sample(monkeypatch) -> None:
    _mock_loader(monkeypatch)
    bands = _s2_bands(10000, include_aerial=True)
    model = TorchGeoDEOBench(bands=bands, mode="s2")
    aerial = torch.full((1, 3, 1, 1), 99.0)
    s2 = torch.tensor(range(1000, 11000, 1000), dtype=torch.float32).view(1, 10, 1, 1)

    normalized = model.normalize_inputs(torch.cat((aerial, s2), dim=1))
    restored = normalized * model._deo_std + model._deo_mean
    expected = torch.cat((torch.tensor([1 / 3, 2 / 3, 1.0]).view(1, 3, 1, 1), s2[:, 3:]), dim=1)
    assert torch.allclose(restored, expected)


def test_deo_s2_reflectance_multispectral_channels_convert_to_dn(monkeypatch) -> None:
    _mock_loader(monkeypatch)
    model = TorchGeoDEOBench(bands=_s2_bands(1), mode="s2")
    reflectance = torch.arange(1, 11, dtype=torch.float32).view(1, 10, 1, 1) / 10

    normalized = model.normalize_inputs(reflectance)
    restored = normalized * model._deo_std + model._deo_mean
    expected = torch.cat((reflectance[:, :3] / 0.3, reflectance[:, 3:] * 10000), dim=1)
    assert torch.allclose(restored, expected)


def test_deo_rejects_bad_mode_normalization_and_missing_s2_band(monkeypatch) -> None:
    _mock_loader(monkeypatch)
    rgb = [_band("rgb", name, 255) for name in ("red", "green", "blue")]
    with pytest.raises(ValueError, match="model_native"):
        TorchGeoDEOBench(bands=rgb, mode="rgb", normalization="identity")
    with pytest.raises(ValueError, match="target_size=224"):
        TorchGeoDEOBench(bands=rgb, mode="rgb", target_size=128)
    with pytest.raises(ValueError, match="missing required"):
        TorchGeoDEOBench(bands=_s2_bands(10000)[:-1], mode="s2")


def test_deo_pools_final_nhwc_map_and_keeps_backbone_frozen(monkeypatch) -> None:
    backbone = _mock_loader(monkeypatch)
    bands = [_band("rgb", name, 255) for name in ("red", "green", "blue")]
    model = TorchGeoDEOBench(bands=bands, mode="rgb")

    features = model.forward_patch_features(torch.ones(2, 3, 17, 19))
    assert features.shape == (2, 1024)
    assert torch.equal(features[0], torch.arange(1024, dtype=features.dtype))
    assert backbone.last_input is not None and backbone.last_input.shape[-2:] == (224, 224)
    assert model.training is False and backbone.training is False
    assert all(not parameter.requires_grad for parameter in backbone.parameters())


def test_deo_rejects_incompatible_checkpoint(monkeypatch) -> None:
    _mock_loader(monkeypatch, incompatible=True)
    with pytest.raises(RuntimeError, match="incompatible"):
        tg_models._load_deo_backbone()


@pytest.mark.parametrize("config_name", ["torchgeo/deo_rgb", "torchgeo/deo_s2"])
def test_deo_configs_compose_without_segmentation(config_name: str) -> None:
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(config_name="config", overrides=[f"model={config_name}"])
    assert cfg.model._target_ == "torchgeo_bench.models.TorchGeoDEOBench"
    assert cfg.model.normalization == "model_native"
    assert "segmentation" not in cfg.model
