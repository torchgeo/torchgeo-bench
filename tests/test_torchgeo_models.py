"""Unit tests for torchgeo wrapper helpers and construction contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from torchvision.transforms.v2 import Normalize as NormalizeV2

from torchgeo_bench.datasets.base import BandSpec, BenchDataset
from torchgeo_bench.datasets.m_eurosat import MEurosat
from torchgeo_bench.datasets.m_so2sat import MSo2Sat
from torchgeo_bench.datasets.resisc45 import RESISC45
from torchgeo_bench.models._input_units import InputUnit
from torchgeo_bench.models.torchgeo_models import (
    _DOFA_SAR_WAVELENGTH_UM,
    TorchGeoCromaBench,
    TorchGeoDOFABench,
    TorchGeoPanopticonBench,
    TorchGeoResNetBench,
    TorchGeoScaleMAEBench,
    TorchGeoSwinBench,
    _adapt_first_conv,
    _extract_normalize_transforms,
    _resolve_dofa_wavelengths,
    _resolve_panopticon_chn_ids,
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


@pytest.mark.parametrize("in_chans", [1, 12])
def test_first_conv_adaptation(in_chans: int):
    model = nn.Sequential(nn.Conv2d(3, 16, 3))
    _adapt_first_conv(model, "0", in_chans=in_chans)
    assert model[0].in_channels == in_chans


@pytest.mark.parametrize("normalize_cls", [Normalize, NormalizeV2])
def test_normalize_transform_extraction(normalize_cls: type[nn.Module]) -> None:
    class _Weights:
        def transforms(self):
            return nn.Sequential(
                nn.Identity(),
                normalize_cls(mean=[0.1, 0.2, 0.3], std=[0.4, 0.5, 0.6]),
                nn.Identity(),
            )

    transform = _extract_normalize_transforms(_Weights())
    assert isinstance(transform, nn.Sequential)
    norm = transform[0]
    assert isinstance(norm, normalize_cls)
    assert tuple(norm.mean) == pytest.approx((0.1, 0.2, 0.3))
    assert tuple(norm.std) == pytest.approx((0.4, 0.5, 0.6))


def test_normalize_transform_none_when_absent():
    class _Weights:
        def transforms(self):
            return nn.Sequential(nn.Identity())

    assert _extract_normalize_transforms(_Weights()) is None


def install_tiny_scalemae_factory(monkeypatch, transforms: nn.Module | None = None):
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _PatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Conv2d(3, 8, 1)

    class _TinyScaleMAE(nn.Module):
        def __init__(self, *, img_size: int, res: float) -> None:
            super().__init__()
            self.patch_embed = _PatchEmbed()
            self.img_size = img_size
            self.res = res

        def forward_features(self, images: torch.Tensor) -> torch.Tensor:
            batch = images.shape[0]
            cls = torch.full((batch, 1, 8), 2.0, device=images.device)
            patches = torch.ones(batch, 4, 8, device=images.device)
            return torch.cat([cls, patches], dim=1)

    calls: list[tuple[dict, _TinyScaleMAE]] = []

    def factory(*, weights, **kwargs):
        del weights
        backbone = _TinyScaleMAE(**kwargs)
        calls.append((kwargs, backbone))
        return backbone

    monkeypatch.setattr(tg_models, "_resolve_torchgeo_factory", lambda _name: factory)
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(
            transforms=lambda: transforms if transforms is not None else nn.Identity()
        ),
    )
    return calls


def test_scalemae_pooling_cls_and_mean(monkeypatch):
    install_tiny_scalemae_factory(monkeypatch)

    bands = _rgb_bands()
    cls_model = TorchGeoScaleMAEBench(
        bands=bands,
        normalization="identity",
        pool="cls",
    )
    mean_model = TorchGeoScaleMAEBench(
        bands=bands,
        normalization="identity",
        pool="mean",
    )
    sample = torch.rand(2, 3, 64, 64)
    cls_out = cls_model.forward_patch_features(sample)
    mean_out = mean_model.forward_patch_features(sample)
    assert cls_out.shape == (2, 8)
    assert mean_out.shape == (2, 8)
    assert torch.allclose(cls_out, torch.full_like(cls_out, 2.0))
    assert torch.allclose(mean_out, torch.full_like(mean_out, 1.0))


def test_scalemae_constructs_selected_grid_without_adapting_rgb_projection(monkeypatch):
    import torchgeo_bench.models.torchgeo_models as tg_models

    calls = install_tiny_scalemae_factory(monkeypatch)
    monkeypatch.setattr(
        tg_models,
        "_adapt_first_conv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not adapt")),
    )
    model = TorchGeoScaleMAEBench(
        bands=_rgb_bands(),
        normalization="identity",
        image_size=64,
        res=3.5,
    )

    kwargs, backbone = calls[0]
    assert kwargs == {"img_size": 64, "res": 3.5}
    assert backbone is model.backbone
    assert model.backbone.patch_embed.proj.in_channels == 3


def test_scalemae_rejects_non_rgb_order() -> None:
    with pytest.raises(ValueError, match="ordered RGB"):
        TorchGeoScaleMAEBench(bands=list(reversed(_rgb_bands())))


def test_scalemae_uses_bandspec_zscore_instead_of_checkpoint_transform(monkeypatch):
    checkpoint_transform = nn.Sequential(
        Normalize(mean=[0.0], std=[10000.0]),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    install_tiny_scalemae_factory(monkeypatch, transforms=checkpoint_transform)
    model = TorchGeoScaleMAEBench(bands=_rgb_bands(), normalization="bandspec_zscore")

    at_dataset_mean = torch.full((1, 3, 8, 8), 1500.0)
    normalized = model.normalize_inputs(at_dataset_mean)
    assert torch.allclose(normalized, torch.zeros_like(normalized))


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

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoResNetBench(bands=_rgb_bands(), normalization="identity")
    out = model.forward_patch_features(torch.rand(2, 3, 64, 64))
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def _sar_band(name: str) -> BandSpec:
    return BandSpec(
        sensor="s1", name=name, source_name=name.upper(), mean=0.0, std=1.0, min=-1.0, max=1.0
    )


def test_dofa_wavelengths_default_sar_bands_to_zhu_xlab_placeholder() -> None:
    """SAR bands (sensor s1/sar) with no wavelength get DOFA's own 3.75um
    placeholder (github.com/zhu-xlab/DOFA waves.json key "2") instead of
    raising, since radar backscatter has no optical wavelength to declare."""
    bands = _s2_multispectral_bands()[:3] + [_sar_band("vh"), _sar_band("vv")]
    wavelengths = _resolve_dofa_wavelengths(bands, None)
    assert wavelengths[-2:] == [_DOFA_SAR_WAVELENGTH_UM, _DOFA_SAR_WAVELENGTH_UM]
    assert wavelengths[:3] == [b.wavelength_um for b in bands[:3]]


def test_torchgeo_backbone_construction_ignores_input_unit_outside_model_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mixed-sensor band sets have no single input unit; must not eagerly call detect_input_unit() outside model_native."""
    import torchgeo_bench.models.torchgeo_models as tg_models

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: nn.Identity()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    bands = _s2_multispectral_bands() + [_sar_band("vh"), _sar_band("vv")]
    for normalization in ("bandspec_zscore", "identity"):
        model = TorchGeoDOFABench(bands=bands, normalization=normalization)
        assert model._dataset_input_unit is None


def test_dofa_wavelengths_still_raises_for_non_sar_missing_wavelength() -> None:
    """A non-SAR band with no wavelength and no known S2 canonical name is a
    real data-declaration gap, not something DOFA has a default for."""
    bad_band = BandSpec(
        sensor="dem", name="elevation", source_name="DEM", mean=0.0, std=1.0, min=0.0, max=1.0
    )
    with pytest.raises(ValueError, match="DOFA wavelengths missing"):
        _resolve_dofa_wavelengths([bad_band], None)


def test_dofa_wavelengths_fall_back_to_s2_table_for_landsat_bands() -> None:
    """m_forestnet.py's Landsat nir/swir_1/swir_2 bands don't set
    wavelength_um -- must fall back to the true Sentinel-2 centre
    wavelength by canonical name rather than raising."""
    from torchgeo_bench.models._band_mapping import S2_WAVELENGTHS_UM

    landsat_bands = [
        BandSpec(
            sensor="landsat", name="nir", source_name="NIR", mean=0.0, std=1.0, min=0.0, max=1.0
        ),
        BandSpec(
            sensor="landsat",
            name="swir_1",
            source_name="SWIR1",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=1.0,
        ),
    ]
    wavelengths = _resolve_dofa_wavelengths(landsat_bands, None)
    assert wavelengths == [S2_WAVELENGTHS_UM["nir"], S2_WAVELENGTHS_UM["swir1"]]


def test_torchgeo_dofa_forward_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyDOFA(nn.Module):
        def forward_features(self, images: torch.Tensor, wavelengths: list[float]) -> torch.Tensor:
            assert len(wavelengths) == images.shape[1]
            return torch.ones(images.shape[0], 8, device=images.device)

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyDOFA()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    model = TorchGeoDOFABench(
        bands=_s2_multispectral_bands(),
        normalization="identity",
    )
    out = model.forward_patch_features(torch.rand(2, 12, 64, 64))
    assert out.ndim == 2
    assert out.shape == (2, 8)
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
    )
    out = model.forward_patch_features(torch.rand(2, 12, 64, 64))
    assert out.ndim == 2
    assert out.shape == (2, 12)
    assert torch.isfinite(out).all()


def test_resolve_panopticon_chn_ids_sar_and_optical() -> None:
    """SAR channels resolve to Panopticon's negative chn_id codes, not a wavelength."""
    bands = _s2_multispectral_bands()[:1] + [_sar_band("vv"), _sar_band("vh")]
    ids = _resolve_panopticon_chn_ids(bands)
    assert ids[0] == pytest.approx(bands[0].wavelength_um * 1000.0)
    assert ids[1:] == [-1.0, -2.0]


def test_resolve_panopticon_chn_ids_raises_for_unknown_polarization() -> None:
    bad = BandSpec(sensor="s1", name="xx", source_name="XX", mean=0.0, std=1.0, min=-1.0, max=1.0)
    with pytest.raises(ValueError, match="chn_ids missing"):
        _resolve_panopticon_chn_ids([bad])


def test_torchgeo_panopticon_model_native_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Panopticon ships no Normalize transform and no fixed pretrain
    mean/std -- it genuinely has no model_native normalization, so asking
    for it must raise rather than silently substituting something else."""
    import torchgeo_bench.models.torchgeo_models as tg_models

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: nn.Identity()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(transforms=nn.Identity()),
    )

    bands = _s2_multispectral_bands()
    native = TorchGeoPanopticonBench(bands=bands, normalization="model_native")

    with pytest.raises(ValueError, match="model_native normalisation is undefined"):
        native.normalize_inputs(torch.rand(2, len(bands), 32, 32) * 5000)


@pytest.mark.parametrize("normalize_cls", [Normalize, NormalizeV2])
def test_channel_mismatch_preserves_tiled_normalize_chain(
    monkeypatch: pytest.MonkeyPatch, normalize_cls: type[nn.Module]
) -> None:
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
            return nn.Sequential(
                normalize_cls(mean=[0.0], std=[2.0]),
                normalize_cls(mean=[1.0, 2.0, 3.0], std=[4.0, 5.0, 6.0]),
            )

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
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
        normalization="model_native",
    )
    normalized = model.normalize_inputs(torch.ones(1, 6, 8, 8))
    expected = torch.tensor([(0.5 - 1.0) / 4.0, (0.5 - 2.0) / 5.0, (0.5 - 3.0) / 6.0]).view(
        1, 3, 1, 1
    )
    assert torch.allclose(normalized[:, :3], expected.expand(1, 3, 8, 8))
    assert torch.allclose(normalized[:, 3:], expected.expand(1, 3, 8, 8))


def test_resnet_can_convert_to_reflectance_before_skipping_weight_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw S2 should skip a checkpoint's uint8 scale before ImageNet z-score."""
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

    class _FakeWeights:
        transforms = nn.Sequential(
            Normalize(mean=[0.0], std=[255.0]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        )

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: _FakeWeights(),
    )

    model = TorchGeoResNetBench(
        bands=_rgb_bands(),
        normalization="model_native",
        normalization_input_unit="reflectance_0_1",
        skip_weight_normalize=1,
    )

    normalized = model.normalize_inputs(torch.full((1, 3, 2, 2), 10000.0))
    expected = torch.tensor([(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225])
    assert torch.allclose(normalized[0, :, 0, 0], expected)


@pytest.mark.parametrize(
    ("dataset_cls", "value"), [(MEurosat, 5000.0), (MSo2Sat, 0.5), (RESISC45, 127.5)]
)
@pytest.mark.parametrize("normalize_cls", [Normalize, NormalizeV2])
def test_resnet_native_normalization_converts_source_units(
    monkeypatch: pytest.MonkeyPatch,
    dataset_cls: type[BenchDataset],
    value: float,
    normalize_cls: type[nn.Module],
) -> None:
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

    transforms = nn.Sequential(
        normalize_cls(mean=[0.0], std=[10000.0]),
        normalize_cls(mean=[0.1, 0.2, 0.3], std=[0.5, 0.25, 0.1]),
    )
    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda *_: SimpleNamespace(transforms=transforms),
    )
    dataset = dataset_cls()
    model = TorchGeoResNetBench(
        bands=dataset.select_band_specs(tuple(dataset.rgb_bands)), normalization="model_native"
    )

    normalized = model.normalize_inputs(torch.full((1, 3, 2, 2), value))
    expected = torch.tensor([(0.5 - 0.1) / 0.5, (0.5 - 0.2) / 0.25, (0.5 - 0.3) / 0.1])
    torch.testing.assert_close(normalized, expected.view(1, 3, 1, 1).expand_as(normalized))
    assert model.expected_input_unit is InputUnit.S2_DN
    assert TorchGeoResNetBench.expected_input_unit is InputUnit.S2_DN


def _dn_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=1200.0,
            std=400.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(3)
    ]


# ---------------------------------------------------------------------------
# _adapt_first_conv fallback path (NotImplementedError from timm)
# ---------------------------------------------------------------------------


def test_adapt_first_conv_fallback_on_timm_not_implemented(monkeypatch):
    """When timm.adapt_input_conv raises NotImplementedError, fallback average replication."""
    from timm.models import _manipulate

    monkeypatch.setattr(
        _manipulate,
        "adapt_input_conv",
        lambda *a, **kw: (_ for _ in ()).throw(NotImplementedError()),
    )
    model = nn.Sequential(nn.Conv2d(13, 16, 3))
    _adapt_first_conv(model, "0", in_chans=3)
    assert model[0].in_channels == 3


def test_adapt_first_conv_noop_same_channels():
    model = nn.Sequential(nn.Conv2d(3, 16, 3))
    original_weight = model[0].weight.data.clone()
    _adapt_first_conv(model, "0", in_chans=3)
    assert torch.equal(model[0].weight.data, original_weight)


# ---------------------------------------------------------------------------
# normalize_inputs: unit conversion only fires for model_native + weights_norm
# ---------------------------------------------------------------------------


def test_normalize_inputs_bandspec_zscore_no_unit_conversion(monkeypatch):
    """bandspec_zscore should NOT apply unit conversion — raw DN values z-scored directly."""
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.pool = nn.AdaptiveAvgPool2d(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(self.conv1(x)).flatten(1)

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_weights", lambda *_: SimpleNamespace(transforms=nn.Identity())
    )

    bands = _dn_bands()  # mean=1200, max=10000
    model = TorchGeoResNetBench(bands=bands, normalization="bandspec_zscore")

    # A tensor at exactly the band mean should z-score to ≈0
    x = torch.full((1, 3, 8, 8), 1200.0)
    normed = model.normalize_inputs(x)
    assert torch.allclose(normed, torch.zeros_like(normed), atol=1e-4)


def test_weights_normalize_only_applies_under_model_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """dataset.normalization must not be silently overridden by the weights transform."""
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinyResNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, 1)
            self.fc = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    class _FakeWeights:
        @staticmethod
        def transforms() -> nn.Sequential:
            return nn.Sequential(Normalize(mean=[0.0, 0.0, 0.0], std=[2.0, 2.0, 2.0]))

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinyResNet()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: _FakeWeights(),
    )

    bands = _dn_bands()
    sample = torch.full((1, 3, 4, 4), 1000.0)

    native = TorchGeoResNetBench(bands=bands, normalization="model_native").normalize_inputs(sample)
    zscore = TorchGeoResNetBench(bands=bands, normalization="bandspec_zscore").normalize_inputs(
        sample
    )
    identity = TorchGeoResNetBench(bands=bands, normalization="identity").normalize_inputs(sample)

    # bandspec_zscore standardises with the BandSpec stats (mean 1200, std 400);
    # identity passes raw values through; only model_native uses the weights'
    # Normalize (std=2 -> 500.0).  Before the fix all three returned 500.0.
    assert torch.allclose(zscore, torch.full_like(zscore, (1000.0 - 1200.0) / 400.0))
    assert torch.allclose(identity, sample)
    assert torch.allclose(native, torch.full_like(native, 500.0))


def test_expected_input_unit_is_derived_per_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Derived units must neither mutate classes nor overwrite declared units."""
    import torchgeo_bench.models.torchgeo_models as tg_models

    class _TinySwin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(nn.Sequential(nn.Conv2d(3, 4, 1)))
            self.head = nn.Identity()

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images.mean(dim=(2, 3))

    class _ReflectanceSwin(TorchGeoSwinBench):
        weights_input_unit = "reflectance_0_1"

    class _DeclaredUnitSwin(TorchGeoSwinBench):
        expected_input_unit = InputUnit.S2_DN

    monkeypatch.setattr(
        tg_models, "_resolve_torchgeo_factory", lambda _name: lambda weights: _TinySwin()
    )
    monkeypatch.setattr(
        tg_models,
        "_resolve_torchgeo_weights",
        lambda _weights_class, _weights_member: SimpleNamespace(
            transforms=nn.Sequential(Normalize(mean=[0.0], std=[255.0]))
        ),
    )

    assert TorchGeoSwinBench.expected_input_unit is None
    model = TorchGeoSwinBench(bands=_rgb_bands(), normalization="model_native")
    assert model.expected_input_unit is InputUnit.UINT8
    assert TorchGeoSwinBench.expected_input_unit is None
    normalized = model.normalize_inputs(torch.full((1, 3, 2, 2), 5000.0))
    torch.testing.assert_close(normalized, torch.full_like(normalized, 0.5))

    reflectance = _ReflectanceSwin(bands=_rgb_bands(), normalization="identity")
    assert reflectance.expected_input_unit is InputUnit.REFLECTANCE_0_1
    assert _ReflectanceSwin.expected_input_unit is None
    assert TorchGeoSwinBench.expected_input_unit is None

    declared = _DeclaredUnitSwin(bands=_rgb_bands(), normalization="identity")
    assert declared.expected_input_unit is InputUnit.S2_DN
    assert _DeclaredUnitSwin.expected_input_unit is InputUnit.S2_DN
