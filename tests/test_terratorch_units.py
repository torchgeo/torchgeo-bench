"""Unit tests for terratorch_models helpers that don't need terratorch installed."""

import pytest
import torch

from torchgeo_bench.models.terratorch_models import _maybe_resize, _reduce_to_vec

# ---------------------------------------------------------------------------
# _maybe_resize
# ---------------------------------------------------------------------------


def test_maybe_resize_none_is_noop():
    x = torch.rand(2, 3, 16, 16)
    out = _maybe_resize(x, size=None)
    assert out is x


@pytest.mark.parametrize(
    ("in_hw", "size"),
    [
        (32, 32),  # same size: no-op
        (16, 32),  # upsample
        (64, 32),  # downsample
    ],
)
def test_maybe_resize(in_hw: int, size: int):
    x = torch.rand(2, 3, in_hw, in_hw)
    out = _maybe_resize(x, size=size)
    assert out.shape == (2, 3, size, size)
    if in_hw == size:
        assert out is x


# ---------------------------------------------------------------------------
# _reduce_to_vec
# ---------------------------------------------------------------------------


def test_reduce_to_vec_4d_mean():
    x = torch.ones(2, 8, 4, 4)
    out = _reduce_to_vec(x, pool="mean")
    assert out.shape == (2, 8)
    assert torch.allclose(out, torch.ones(2, 8))


def test_reduce_to_vec_4d_both_doubles_dim():
    x = torch.rand(2, 8, 4, 4)
    out = _reduce_to_vec(x, pool="both")
    assert out.shape == (2, 16)


def test_reduce_to_vec_3d_mean():
    x = torch.rand(2, 10, 8)  # (B, T, C)
    out = _reduce_to_vec(x, pool="mean")
    assert out.shape == (2, 8)


def test_reduce_to_vec_list_takes_last():
    a = torch.rand(2, 4, 2, 2)
    b = torch.rand(2, 8, 2, 2)
    out = _reduce_to_vec([a, b], pool="mean")
    # last element is b, shape (2, 8, 2, 2) → GAP → (2, 8)
    assert out.shape == (2, 8)


def test_reduce_to_vec_2d_passthrough():
    x = torch.rand(2, 16)
    out = _reduce_to_vec(x, pool="mean")
    assert out is x
