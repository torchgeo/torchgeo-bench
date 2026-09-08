"""Tests for the :class:`BenchModel` interface."""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models.interface import BenchModel


def _bands(n: int = 2) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=float(10 * (i + 1)),
            std=float(2 * (i + 1)),
            min=0.0,
            max=255.0,
        )
        for i in range(n)
    ]


class _Toy(BenchModel):
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        return images.flatten(1)[:, :4]


def test_default_zscore_normalization():
    m = _Toy(bands=_bands(2))
    x = torch.tensor([[[[12.0]], [[24.0]]]], dtype=torch.float32)
    y = m.normalize_inputs(x)
    # Both inputs are one standard deviation above their band means.
    assert torch.allclose(y, torch.ones_like(y), atol=1e-6)


def test_template_method_calls_normalize(monkeypatch):
    """The public forward path must normalize inputs exactly once."""
    m = _Toy(bands=_bands(2))
    calls: list[torch.Tensor] = []

    def spy(images: torch.Tensor) -> torch.Tensor:
        calls.append(images)
        return images

    monkeypatch.setattr(m, "normalize_inputs", spy)
    x = torch.zeros((1, 2, 4, 4))
    _ = m(x)
    assert len(calls) == 1
    assert calls[0] is x


def test_normalize_inputs_buffer_dtype():
    m = _Toy(bands=_bands(2))
    x16 = torch.zeros((1, 2, 1, 1), dtype=torch.float16)
    y = m.normalize_inputs(x16)
    assert y.dtype == torch.float16


def test_empty_bands_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        _Toy(bands=[])
