"""Unit tests for terratorch_models helpers that don't need terratorch installed."""

import pytest
import torch
import torch.nn.functional as F

from torchgeo_bench.models.terratorch_models import _maybe_resize, _reduce_to_vec


# ---------------------------------------------------------------------------
# _maybe_resize
# ---------------------------------------------------------------------------


def test_maybe_resize_none_is_noop():
    x = torch.rand(2, 3, 16, 16)
    out = _maybe_resize(x, size=None)
    assert out is x


def test_maybe_resize_same_size_is_noop():
    x = torch.rand(2, 3, 32, 32)
    out = _maybe_resize(x, size=32)
    assert out is x


def test_maybe_resize_upsample():
    x = torch.rand(2, 3, 16, 16)
    out = _maybe_resize(x, size=32)
    assert out.shape == (2, 3, 32, 32)


def test_maybe_resize_downsample():
    x = torch.rand(2, 3, 64, 64)
    out = _maybe_resize(x, size=32)
    assert out.shape == (2, 3, 32, 32)


# ---------------------------------------------------------------------------
# _reduce_to_vec
# ---------------------------------------------------------------------------


def test_reduce_to_vec_4d_mean():
    x = torch.ones(2, 8, 4, 4)
    out = _reduce_to_vec(x, pool="mean")
    assert out.shape == (2, 8)
    assert torch.allclose(out, torch.ones(2, 8))


def test_reduce_to_vec_4d_cls_acts_as_gap():
    x = torch.ones(2, 8, 4, 4) * 3.0
    out = _reduce_to_vec(x, pool="cls")
    assert out.shape == (2, 8)


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


# ---------------------------------------------------------------------------
# _build_backbone: import error when terratorch not installed
# ---------------------------------------------------------------------------


def test_build_backbone_missing_terratorch(monkeypatch):
    import builtins
    from torchgeo_bench.models.terratorch_models import _build_backbone

    real_import = builtins.__import__

    def _mock(name, *a, **kw):
        if "terratorch" in name:
            raise ImportError("mocked missing terratorch")
        return real_import(name, *a, **kw)

    with pytest.raises(ImportError, match="terratorch is required"):
        monkeypatch.setattr(builtins, "__import__", _mock)
        _build_backbone("some_backbone")
