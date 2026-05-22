"""Tests for _normalization: build_normalizer strategies."""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._input_units import InputUnit
from torchgeo_bench.models._normalization import build_normalizer


def _bands(
    maxvals: list[float], means: list[float] | None = None, stds: list[float] | None = None
) -> list[BandSpec]:
    n = len(maxvals)
    means = means or [m / 2 for m in maxvals]
    stds = stds or [m / 4 for m in maxvals]
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=means[i],
            std=stds[i],
            min=0.0,
            max=maxvals[i],
        )
        for i in range(n)
    ]


def test_identity_is_noop():
    bands = _bands([10000.0, 10000.0])
    fn = build_normalizer("identity", bands)
    x = torch.randn(2, 2, 4, 4) * 5000
    out = fn(x)
    assert out is x


def test_bandspec_zscore_zero_mean():
    """After z-scoring with exact band mean, output mean should be ~0."""
    means = [1000.0, 2000.0]
    stds = [500.0, 800.0]
    bands = _bands([10000.0, 10000.0], means=means, stds=stds)
    fn = build_normalizer("bandspec_zscore", bands)
    # constant images equal to the mean
    x = torch.zeros(1, 2, 8, 8)
    x[0, 0] = means[0]
    x[0, 1] = means[1]
    out = fn(x)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)


def test_bandspec_zscore_unit_variance():
    bands = _bands([10000.0], means=[500.0], stds=[250.0])
    fn = build_normalizer("bandspec_zscore", bands)
    torch.manual_seed(0)
    x = torch.randn(100, 1, 1, 1) * 250 + 500
    out = fn(x)
    assert abs(out.mean().item()) < 0.1
    assert abs(out.std().item() - 1.0) < 0.15


def test_minmax_range():
    """MINMAX should map [min, max] → [0, 1]."""
    bands = [BandSpec(sensor="s2", name="b", source_name="B", mean=5.0, std=2.0, min=2.0, max=12.0)]
    fn = build_normalizer("minmax", bands)
    x_min = torch.tensor([[[[2.0]]]])
    x_max = torch.tensor([[[[12.0]]]])
    assert torch.allclose(fn(x_min), torch.zeros(1, 1, 1, 1), atol=1e-6)
    assert torch.allclose(fn(x_max), torch.ones(1, 1, 1, 1), atol=1e-6)


def test_minmax_zscore_produces_finite():
    bands = _bands([1.0, 1.0], means=[0.3, 0.5], stds=[0.1, 0.2])
    fn = build_normalizer("minmax_zscore", bands)
    x = torch.rand(4, 2, 8, 8)
    out = fn(x)
    assert torch.isfinite(out).all()
    assert out.shape == x.shape


def test_model_native_s2dn_to_reflectance():
    """model_native with S2 DN bands + REFLECTANCE expected unit → divides by 10000."""
    bands = _bands([10000.0])
    fn = build_normalizer("model_native", bands, expected_input_unit=InputUnit.REFLECTANCE_0_1)
    x = torch.tensor([[[[10000.0]]]])
    out = fn(x)
    assert torch.allclose(out, torch.ones(1, 1, 1, 1), atol=1e-5)


def test_model_native_with_pretrain_stats():
    """model_native with pretrain_mean/std applies affine after unit conversion."""
    bands = _bands([10000.0])
    fn = build_normalizer(
        "model_native",
        bands,
        expected_input_unit=InputUnit.REFLECTANCE_0_1,
        pretrain_mean=[0.5],
        pretrain_std=[0.5],
    )
    x = torch.tensor([[[[10000.0]]]])  # → /10000 → 1.0 → (1.0 - 0.5) / 0.5 = 1.0
    out = fn(x)
    assert torch.allclose(out, torch.ones(1, 1, 1, 1), atol=1e-5)


def test_model_native_requires_expected_unit():
    bands = _bands([10000.0])
    with pytest.raises(ValueError, match="expected_input_unit"):
        build_normalizer("model_native", bands)


def test_model_native_s2dn_target():
    """model_native with S2_DN target is a no-op on DN data."""
    bands = _bands([10000.0])
    fn = build_normalizer("model_native", bands, expected_input_unit=InputUnit.S2_DN)
    x = torch.tensor([[[[5000.0]]]])
    out = fn(x)
    assert torch.allclose(out, x, atol=1e-5)
