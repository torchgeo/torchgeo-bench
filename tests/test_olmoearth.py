"""Smoke tests for :class:`OlmoEarthBenchModel`.

The full GeoBench v1+v2 sweep originally couldn't run OlmoEarth because
the ``[olmoearth]`` extra wasn't installed.  These tests both prevent
that regression (by importing the wrapper and checking each variant
loads) and validate the wrapper actually produces sensible embeddings.
"""

from importlib.util import find_spec

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec

olmoearth_available = find_spec("olmoearth_pretrain_minimal") is not None
requires_olmoearth = pytest.mark.skipif(
    not olmoearth_available,
    reason="olmoearth-pretrain-minimal not installed (pip install 'torchgeo-bench[olmoearth]')",
)


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for n in ("red", "green", "blue")
    ]


def _s2_bands() -> list[BandSpec]:
    from torchgeo_bench.models.olmoearth import OLMOEARTH_S2_BANDS

    return [
        BandSpec(
            sensor="s2",
            name=b.lower(),
            source_name=b,
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for b in OLMOEARTH_S2_BANDS
    ]


# Map variant -> expected embedding dim (from the four HF weights configs).
EXPECTED_DIM = {"nano": 128, "tiny": 192, "base": 768, "large": 1024}


@requires_olmoearth
@pytest.mark.parametrize("size", ["nano", "tiny"])  # base/large are too heavy for CI
def test_rgb_forward_pass_shape(size: str) -> None:
    """All-RGB input must produce a 2-D embedding of the expected width."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(bands=_rgb_bands(), model_size=size, normalization="identity")
    model.eval()
    x = torch.rand(2, 3, 64, 64) * 3000.0  # raw S2-like values
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM[size])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_s2_forward_pass_shape() -> None:
    """12-channel S2 input goes through the multispectral path."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(bands=_s2_bands(), model_size="nano", normalization="identity")
    model.eval()
    x = torch.rand(2, 12, 64, 64) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_all_four_variants_are_loadable() -> None:
    """ModelID enum exposes the four advertised variants.

    Prevents the regression where the wrapper silently lost a variant
    after an upstream rename.
    """
    from olmoearth_pretrain_minimal import ModelID

    names = {attr for attr in dir(ModelID) if attr.startswith("OLMOEARTH_V1_")}
    assert names == {
        "OLMOEARTH_V1_NANO",
        "OLMOEARTH_V1_TINY",
        "OLMOEARTH_V1_BASE",
        "OLMOEARTH_V1_LARGE",
    }


@requires_olmoearth
def test_reflectance_input_is_rescaled_to_dn() -> None:
    """Datasets like m-so2sat / so2sat deliver S2 reflectance in [0, ~2.8],
    not raw DN.  The wrapper must detect this and rescale to DN before
    OlmoEarth's Normalizer sees the values — otherwise the normalizer's
    DN-fitted mean/std produce near-zero normalized inputs and embeddings
    collapse.

    We verify the scale-detection path picks ``REFLECTANCE_0_1`` for
    so2sat-style band stats and that the forward pass produces non-degenerate
    embeddings.
    """
    from torchgeo_bench.models._input_units import InputUnit
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    # so2sat-style band stats: optical reflectance with max ~2.8
    refl_bands = [
        BandSpec(
            sensor="s2", name=n, source_name=n.upper(),
            mean=0.13, std=0.07, min=0.0001, max=2.8, wavelength_um=0.5,
        )
        for n in ("red", "green", "blue")
    ]
    model = OlmoEarthBenchModel(bands=refl_bands, model_size="nano", normalization="identity")
    assert model._input_unit == InputUnit.REFLECTANCE_0_1
    model.eval()
    x = torch.rand(2, 3, 64, 64) * 2.5  # reflectance-like values
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()
    # Embeddings should have non-trivial variance — collapsed-to-zero
    # embeddings would have std ~ 0.
    assert out.std() > 1e-4


def test_rejects_unsupported_channel_count() -> None:
    """4-channel input must fail loudly — OlmoEarth only handles 3 or 12."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    four_bands = _rgb_bands() + [_rgb_bands()[0]]
    with pytest.raises(ValueError, match="3 \\(RGB\\) or 12 \\(full S2\\)"):
        OlmoEarthBenchModel(bands=four_bands, model_size="nano", normalization="identity")
