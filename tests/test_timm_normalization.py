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
    assert cfg["mean"] is not None
    assert cfg["std"] is not None

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
    timm's pretrained_cfg."""
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


def test_minmax_zscore_uses_actual_bandspec_stats():
    """MINMAX_ZSCORE must derive post-minmax mean/std from BandSpec, not assume 0.5/0.25."""
    from torchgeo_bench.datasets.base import BandSpec
    from torchgeo_bench.models._normalization import NormalizationStrategy, build_normalizer

    # band: min=0, max=10, mean=3, std=2  => post-minmax mean=0.3, std=0.2
    band = BandSpec(sensor="test", name="b", source_name="B", mean=3.0, std=2.0, min=0.0, max=10.0)
    norm = build_normalizer(NormalizationStrategy.MINMAX_ZSCORE, [band])

    x = torch.tensor([[[[3.0]]]])  # raw value == bandspec mean
    out = norm(x)
    # after minmax: (3-0)/10 = 0.3; after zscore: (0.3 - 0.3) / 0.2 = 0.0
    assert abs(out.item()) < 1e-5, f"expected ~0 at band mean, got {out.item()}"

    x_max = torch.tensor([[[[10.0]]]])
    out_max = norm(x_max)
    # after minmax: 1.0; after zscore: (1.0 - 0.3) / 0.2 = 3.5
    assert abs(out_max.item() - 3.5) < 1e-4, f"expected 3.5 at band max, got {out_max.item()}"


def _band(name: str, wavelength_um: float | None) -> "BandSpec":
    from torchgeo_bench.datasets.base import BandSpec

    return BandSpec(
        sensor="s2",
        name=name,
        source_name=name.upper(),
        mean=0.0,
        std=1.0,
        min=0.0,
        max=1.0,
        wavelength_um=wavelength_um,
    )


def test_rgb_first_permutation_reorders_bgr_native_bands():
    """Sentinel-2's native declaration order (blue, green, red, ...) must be
    reordered so the pretrained RGB kernel's R/G/B weights line up with the
    physically-matching bands, identified by wavelength (name-agnostic)."""
    from torchgeo_bench.models.timm import _rgb_first_permutation

    # Native S2 order: b02=blue, b03=green, b04=red, b08=nir
    bands = [_band("b02", 0.49), _band("b03", 0.56), _band("b04", 0.665), _band("b08", 0.842)]
    perm = _rgb_first_permutation(bands)
    assert perm == [2, 1, 0, 3]  # red, green, blue, nir


def test_rgb_first_permutation_none_without_full_triplet():
    """SAR-only or otherwise non-optical band sets have nothing to reorder."""
    from torchgeo_bench.models.timm import _rgb_first_permutation

    bands = [_band("vv", None), _band("vh", None)]
    assert _rgb_first_permutation(bands) is None

    # Missing the red band entirely.
    bands = [_band("b02", 0.49), _band("b03", 0.56), _band("nir", 0.842)]
    assert _rgb_first_permutation(bands) is None


def test_multichannel_pretrained_permutes_channels_before_backbone():
    """Pretrained + multi-channel must feed the backbone R/G/B-first, even
    when the dataset declares bands in native blue/green/red/... order."""
    from torchgeo_bench.models.timm import TimmPatchBenchModel

    bands = [_band("b02", 0.49), _band("b03", 0.56), _band("b04", 0.665), _band("b08", 0.842)]
    model = TimmPatchBenchModel(bands=bands, model_name="resnet18", pretrained=True)
    assert model._channel_perm == [2, 1, 0, 3]

    seen = {}

    class _Capture(torch.nn.Module):
        def forward(self, images):
            seen["images"] = images
            return torch.zeros(images.shape[0], 8)

    model.backbone = _Capture()
    raw = torch.arange(4).float().view(1, 4, 1, 1).expand(1, 4, 2, 2).clone()
    model._forward_patch_features(raw)
    # channel 0 (native blue=1.0) must now carry channel-index-2's original
    # value (0.0, native red) after the R/G/B-first permutation, etc.
    assert torch.equal(seen["images"][0, :, 0, 0], torch.tensor([2.0, 1.0, 0.0, 3.0]))


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
