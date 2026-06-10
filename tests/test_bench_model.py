"""Unit tests for :class:`BenchModel` ABC contract."""

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
    """Per-channel z-score uses BandSpec.{mean, std}."""
    m = _Toy(bands=_bands(2))
    x = torch.tensor([[[[12.0]], [[24.0]]]], dtype=torch.float32)  # (1, 2, 1, 1)
    y = m.normalize_inputs(x)
    # band 0: mean=10, std=2  → (12-10)/2 = 1
    # band 1: mean=20, std=4  → (24-20)/4 = 1
    assert torch.allclose(y, torch.ones_like(y), atol=1e-6)


def test_template_method_calls_normalize(monkeypatch):
    """`forward_patch_features` always routes through `normalize_inputs`."""
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
    """Buffers are recast to input dtype so fp16 / bf16 inputs work."""
    m = _Toy(bands=_bands(2))
    x16 = torch.zeros((1, 2, 1, 1), dtype=torch.float16)
    y = m.normalize_inputs(x16)
    assert y.dtype == torch.float16


def test_empty_bands_rejected():
    """Constructing with no bands is a clear configuration error."""
    with pytest.raises(ValueError, match="non-empty"):
        _Toy(bands=[])


def test_num_channels_property():
    """`num_channels` is derived from `len(bands)`."""
    m = _Toy(bands=_bands(5))
    assert m.num_channels == 5
