"""Unit tests for torchgeo wrapper helpers and construction contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchvision.transforms import Normalize

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models.torchgeo_models import (
    TorchGeoCromaBench,
    TorchGeoDOFABench,
    TorchGeoEarthLocBench,
    TorchGeoPanopticonBench,
    TorchGeoResNetBench,
    TorchGeoScaleMAEBench,
    TorchGeoSwinBench,
    _adapt_first_conv,
    _extract_normalize_transforms,
    _resolve_torchgeo_factory,
    _resolve_torchgeo_weights,
)


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for name in ("red", "green", "blue")
    ]


def _s2_multispectral_bands() -> list[BandSpec]:
    names = [
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
    wavelengths = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.61, 2.19]
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=0.2,
            std=0.05,
            min=0.0,
            max=1.0,
            wavelength_um=wavelength,
        )
        for name, wavelength in zip(names, wavelengths, strict=True)
    ]


def test_factory_resolution_failure():
    with pytest.raises(ValueError, match="factory function"):
        _resolve_torchgeo_factory("torchgeo.models.NotARealModel")


def test_weights_resolution_failure(monkeypatch):
    class _FakeWeights:
        REAL = object()

    import torchgeo_bench.models.torchgeo_models as tg_models

    monkeypatch.setattr(tg_models.tgm, "FakeWeights", _FakeWeights, raising=False)
    with pytest.raises(ValueError, match="has no member"):
        _resolve_torchgeo_weights("FakeWeights", "FAKE_MEMBER")


def test_first_conv_adaptation_single_channel():
    model = nn.Sequential(nn.Conv2d(3, 16, 3))
    _adapt_first_conv(model, "0", in_chans=1)
    assert model[0].in_channels == 1


def test_first_conv_adaptation_multichannel():
    model = nn.Sequential(nn.Conv2d(3, 16, 3))
    _adapt_first_conv(model, "0", in_chans=12)
    assert model[0].in_channels == 12


def test_normalize_transform_extraction():
    class _Weights:
        def transforms(self):
            return nn.Sequential(
                nn.Identity(),
                Normalize(mean=[0.1, 0.2, 0.3], std=[0.4, 0.5, 0.6]),
                nn.Identity(),
            )

    transform = _extract_normalize_transforms(_Weights())
    assert isinstance(transform, nn.Sequential)
    norm = transform[0]
    assert isinstance(norm, Normalize)
    assert tuple(norm.mean) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(norm.std) == pytest.approx((0.4, 0.5, 0.6))


def test_normalize_transform_none_when_absent():
    class _Weights:
        def transforms(self):
            return nn.Sequential(nn.Identity())

    assert _extract_normalize_transforms(_Weights()) is None


def test_scalemae_pooling_cls_and_mean(monkeypatch):
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(3, 8, 1)

    class _TinyScaleMAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = _PatchEmbed()

        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            batch = images.shape[0]
            cls = torch.full((batch, 1, 8), 2.0, device=images.device)
            patches = torch.ones(batch, 4, 8, device=images.device)
            return torch.cat([cls, patches], dim=1)

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyScaleMAE())
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    bands = _rgb_bands()
    cls_model = TorchGeoScaleMAEBench(
        bands=bands,
        normalization="identity",
        input_unit_check="ignore",
        pool="cls",
    )
    mean_model = TorchGeoScaleMAEBench(
        bands=bands,
        normalization="identity",
        input_unit_check="ignore",
        pool="mean",
    )
    sample = torch.rand(2, 3, 64, 64)
    cls_out = cls_model.forward_patch_features(sample)
    mean_out = mean_model.forward_patch_features(sample)
    assert cls_out.shape == (2, 8)
    assert mean_out.shape == (2, 8)
    assert torch.allclose(cls_out, torch.full_like(cls_out, 2.0))
    assert torch.allclose(mean_out, torch.full_like(mean_out, 1.0))


def test_torchgeo_resnet_forward_shape(monkeypatch):
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            feats = self.pool(self.conv1(images))
            return feats.flatten(1)

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet())
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoResNetBench(bands=_rgb_bands(), normalization="identity", input_unit_check="ignore")
    out = model.forward_patch_features(torch.rand(2, 3, 64, 64))
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_torchgeo_swin_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinySwin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(nn.Sequential(nn.Conv2d(3, 8, 1)))
            self.head = nn.Linear(8, 8)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            feats = self.pool(self.features[0][0](images)).flatten(1)
            return feats

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinySwin())
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoSwinBench(
        bands=_rgb_bands(),
        normalization="identity",
        input_unit_check="ignore",
    )
    out = model.forward_patch_features(torch.rand(2, 3, 64, 64))
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_torchgeo_dofa_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyDOFA(nn.Module):
        def forward_features(self, images: torch.Tensor, wavelengths: list[float]) -> torch.Tensor:
            assert len(wavelengths) == images.shape[1]
            return torch.ones(images.shape[0], 8, device=images.device)

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyDOFA())
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoDOFABench(
        bands=_s2_multispectral_bands(),
        normalization="identity",
        input_unit_check="ignore",
    )
    out = model.forward_patch_features(torch.rand(2, 12, 64, 64))
    assert out.ndim == 2
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_torchgeo_earthloc_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyEarthLoc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential()
            self.backbone.conv1 = nn.Conv2d(3, 6, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            feats = self.pool(self.backbone.conv1(images))
            return feats.flatten(1)

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyEarthLoc()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoEarthLocBench(
        bands=_rgb_bands(),
        normalization="identity",
        input_unit_check="ignore",
    )
    out = model.forward_patch_features(torch.rand(2, 3, 64, 64))
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_torchgeo_croma_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyCroma(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn_bias = torch.zeros(1)

        def s2_encoder(self, imgs: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
            del attn_bias
            batch = imgs.shape[0]
            return torch.ones(batch, 4, 8, device=imgs.device)

        def s2_GAP_FFN(self, x: torch.Tensor) -> torch.Tensor:
            return x

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyCroma()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoCromaBench(
        bands=_s2_multispectral_bands(),
        normalization="identity",
        input_unit_check="ignore",
    )
    out = model.forward_patch_features(torch.rand(2, 12, 64, 64))
    assert out.ndim == 2
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_torchgeo_panopticon_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyPanopticon(nn.Module):
        def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            imgs = batch["imgs"]
            chn_ids = batch["chn_ids"]
            assert chn_ids.shape[0] == imgs.shape[0]
            return imgs.mean(dim=(2, 3))

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyPanopticon()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoPanopticonBench(
        bands=_s2_multispectral_bands(),
        normalization="identity",
        input_unit_check="ignore",
    )
    out = model.forward_patch_features(torch.rand(2, 12, 64, 64))
    assert out.ndim == 2
    assert out.shape == (2, 12)
    assert torch.isfinite(out).all()


def test_channel_mismatch_uses_tiled_normalize(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.pool(self.conv1(images)).flatten(1)

    class _FakeWeights:
        @staticmethod
        def transforms() -> nn.Sequential:
            return nn.Sequential(Normalize(mean=[1.0, 2.0, 3.0], std=[4.0, 5.0, 6.0]))

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet())
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: _FakeWeights(),
    )

    six_bands = [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=1000.0,
            std=100.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(6)
    ]
    model = TorchGeoResNetBench(
        bands=six_bands,
        normalization="identity",
        input_unit_check="ignore",
    )
    normalized = model.normalize_inputs(torch.ones(1, 6, 8, 8))
    expected = torch.tensor([(1.0 - 1.0) / 4.0, (1.0 - 2.0) / 5.0, (1.0 - 3.0) / 6.0]).view(
        1, 3, 1, 1
    )
    assert torch.allclose(normalized[:, :3], expected.expand(1, 3, 8, 8))
