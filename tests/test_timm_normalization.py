"""Unit tests for :class:`TimmPatchBenchModel.normalize_inputs`."""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec


def _rgb_bands(*, mins=(0.0, 0.0, 0.0), maxs=(28000.0, 28000.0, 28000.0)) -> list[BandSpec]:
    """Build a 3-band BandSpec list mimicking S2-style raw RGB ranges."""
    names = ("red", "green", "blue")
    return [
        BandSpec(
            sensor="s2",
            name=names[i],
            source_name=names[i].upper(),
            mean=float(maxs[i] / 2),
            std=float(maxs[i] / 4),
            min=float(mins[i]),
            max=float(maxs[i]),
        )
        for i in range(3)
    ]


@pytest.fixture(autouse=True)
def _block_pretrained_download(monkeypatch):
    """Force ``pretrained=False`` so tests don't hit Hugging Face."""
    import timm

    real_create = timm.create_model

    def _no_pretrained(*args, **kwargs):
        kwargs["pretrained"] = False
        return real_create(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", _no_pretrained)


def test_imagenet_normalization_rescales_raw_values_to_unit_interval():
    """``imagenet`` mode must min-max scale to [0, 1] using BandSpec stats before mean/std."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(28000.0, 28000.0, 28000.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="imagenet",
    )

    # Pixel value at half of each band's max range -> 0.5 in [0, 1] -> (0.5 - mean) / std
    raw = torch.full((1, 3, 4, 4), 14000.0)
    out = model.normalize_inputs(raw)

    expected = (
        torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)
    # Sanity: every channel should be roughly O(1), not O(thousands).
    assert out.abs().max() < 5.0, (
        f"normalized output should be O(1) but got max |x| = {out.abs().max().item():.2f} — "
        "this is the bug the fix addresses."
    )


def test_imagenet_normalization_band_min_subtracted_first():
    """When BandSpec.min > 0 the rescale must subtract band_min before dividing by range."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    bands = _rgb_bands(mins=(100.0, 100.0, 100.0), maxs=(900.0, 900.0, 900.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="imagenet",
    )

    raw = torch.full((1, 3, 2, 2), 500.0)  # midpoint -> 0.5 after rescale
    out = model.normalize_inputs(raw)

    expected = (
        torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    ) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_imagenet_normalization_rejects_non_rgb():
    """``imagenet`` mode must refuse to instantiate with a non-3-channel band list."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    bands = _rgb_bands() + [_rgb_bands()[0]]  # 4 bands
    with pytest.raises(ValueError, match="requires 3 input channels"):
        TimmPatchBenchModel(
            bands=bands,
            model_name="resnet18",
            pretrained=False,
            input_normalization="imagenet",
        )


def test_timm_default_normalization_uses_default_cfg_stats():
    """``timm_default`` must read mean/std from the backbone's default_cfg."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    bands = _rgb_bands()
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="timm_default",
    )

    cfg = model.backbone.default_cfg
    assert cfg["mean"] is not None and cfg["std"] is not None

    raw = torch.full((1, 3, 2, 2), 14000.0)  # midpoint of 0..28000
    out = model.normalize_inputs(raw)
    expected = (
        torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1) - torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    ) / torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_bands_zscore_unaffected_by_imagenet_path():
    """``bands_zscore`` mode must still use BandSpec.{mean, std}, not RGB stats."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(28000.0, 28000.0, 28000.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        input_normalization="bands_zscore",
    )
    raw = torch.full((1, 3, 2, 2), 14000.0)  # equal to BandSpec.mean for each channel
    out = model.normalize_inputs(raw)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


def test_none_normalization_is_identity():
    """``none`` mode passes inputs through untouched."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    model = TimmPatchBenchModel(
        bands=_rgb_bands(),
        model_name="resnet18",
        pretrained=False,
        input_normalization="none",
    )
    raw = torch.tensor([[[[1234.0]], [[5678.0]], [[9012.0]]]])
    out = model.normalize_inputs(raw)
    assert torch.equal(out, raw)


def test_model_native_uses_timm_pretrained_stats():
    """``normalization=model_native`` on a timm wrapper pulls mean/std from
    timm's pretrained_cfg (previously raised ``requires expected_input_unit``
    because TimmPatchBenchModel didn't populate the BenchModel-level stats).
    """
    import timm

    from torchgeo_bench.models.timm import TimmPatchBenchModel

    cfg = timm.get_pretrained_cfg("resnet18")
    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(1.0, 1.0, 1.0))  # already in [0, 1]
    model = TimmPatchBenchModel(
        bands=bands,
        model_name="resnet18",
        pretrained=False,
        normalization="model_native",
    )
    raw = torch.zeros(1, 3, 2, 2)  # zero reflectance -> -mean/std after normalize
    out = model.normalize_inputs(raw)
    expected = -torch.tensor(cfg.mean).view(1, 3, 1, 1) / torch.tensor(cfg.std).view(1, 3, 1, 1)
    assert torch.allclose(out, expected.expand_as(out), atol=1e-5)


def test_model_native_picks_up_variant_stats():
    """A timm variant (e.g. dinov3.sat493m) ships its own pretrained stats —
    the wrapper must use those, not the generic ImageNet defaults."""
    import timm

    from torchgeo_bench.models.timm import TimmPatchBenchModel

    name = "vit_large_patch16_dinov3.sat493m"
    cfg = timm.get_pretrained_cfg(name)
    assert tuple(cfg.mean) != (0.485, 0.456, 0.406), "stats should differ from ImageNet"

    bands = _rgb_bands(mins=(0.0, 0.0, 0.0), maxs=(1.0, 1.0, 1.0))
    model = TimmPatchBenchModel(
        bands=bands,
        model_name=name,
        pretrained=False,
        normalization="model_native",
    )
    assert model.pretrain_mean == list(cfg.mean)
    assert model.pretrain_std == list(cfg.std)


def test_minmax_zscore_uses_actual_bandspec_stats():
    """MINMAX_ZSCORE must derive post-minmax mean/std from BandSpec, not assume 0.5/0.25."""
    from torchgeo_bench.datasets.base import BandSpec
    from torchgeo_bench.models._normalization import NormalizationStrategy, build_normalizer

    # band: min=0, max=10, mean=3, std=2  => post-minmax mean=0.3, std=0.2
    band = BandSpec(
        sensor="test", name="b", source_name="B", mean=3.0, std=2.0, min=0.0, max=10.0
    )
    norm = build_normalizer(NormalizationStrategy.MINMAX_ZSCORE, [band])

    x = torch.tensor([[[[3.0]]]])  # raw value == bandspec mean
    out = norm(x)
    # after minmax: (3-0)/10 = 0.3; after zscore: (0.3 - 0.3) / 0.2 = 0.0
    assert abs(out.item()) < 1e-5, f"expected ~0 at band mean, got {out.item()}"

    x_max = torch.tensor([[[[10.0]]]])
    out_max = norm(x_max)
    # after minmax: 1.0; after zscore: (1.0 - 0.3) / 0.2 = 3.5
    assert abs(out_max.item() - 3.5) < 1e-4, f"expected 3.5 at band max, got {out_max.item()}"


def test_unknown_timm_model_name_raises_clearly():
    """A typo in ``model_name`` must fail loudly at construction (we don't
    silently swallow the missing pretrained_cfg)."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    with pytest.raises(RuntimeError, match="no pretrained cfg"):
        TimmPatchBenchModel(
            bands=_rgb_bands(),
            model_name="this_model_definitely_does_not_exist_xyz",
            pretrained=False,
        )
