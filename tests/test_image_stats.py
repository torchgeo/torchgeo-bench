"""Tests for ImageStatsBench model."""

import numpy as np
import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models.image_stats import ImageStatsBench


def _bands(n: int = 4) -> list[BandSpec]:
    return [
        BandSpec(sensor="s2", name=f"b{i}", source_name=f"B{i}", mean=500.0, std=200.0, min=0.0, max=10000.0)
        for i in range(n)
    ]


def test_normalize_inputs_is_identity():
    """normalize_inputs must return the tensor unchanged."""
    model = ImageStatsBench(bands=_bands(3))
    x = torch.randn(2, 3, 16, 16) * 5000
    out = model.normalize_inputs(x)
    assert out is x


def test_output_shape():
    """Output should be (B, 4*C) — mean/std/max/min per channel."""
    n_bands = 4
    model = ImageStatsBench(bands=_bands(n_bands))
    x = torch.rand(3, n_bands, 8, 8) * 10000
    feats = model(x)
    assert feats.shape == (3, 4 * n_bands)


def test_output_stats_values():
    """Verify mean/std/max/min slices match manual computation."""
    n = 2
    model = ImageStatsBench(bands=_bands(n))
    # Constant-value images per channel so stats are predictable
    x = torch.zeros(1, n, 4, 4)
    x[0, 0] = 3.0
    x[0, 1] = 7.0
    feats = model(x)
    # mean
    assert torch.allclose(feats[0, 0], torch.tensor(3.0))
    assert torch.allclose(feats[0, 1], torch.tensor(7.0))
    # std of constant = 0
    assert torch.allclose(feats[0, n], torch.tensor(0.0), atol=1e-5)
    # max = min = constant
    assert torch.allclose(feats[0, 2 * n], torch.tensor(3.0))
    assert torch.allclose(feats[0, 3 * n], torch.tensor(3.0))


def test_single_pixel_image():
    """1×1 spatial images produce correct mean/max/min (std is NaN for n=1 — expected)."""
    model = ImageStatsBench(bands=_bands(2))
    x = torch.tensor([[[[5.0]], [[9.0]]]])  # (1, 2, 1, 1)
    feats = model(x)
    assert feats.shape == (1, 8)
    # mean and max/min are finite; std with n=1 is NaN by design
    assert torch.isfinite(feats[0, 0])  # mean ch0
    assert torch.isfinite(feats[0, 1])  # mean ch1
    assert torch.isfinite(feats[0, 4])  # max ch0
    assert torch.isfinite(feats[0, 6])  # min ch0
