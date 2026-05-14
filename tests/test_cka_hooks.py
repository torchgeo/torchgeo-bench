import numpy as np
import torch
import torch.nn as nn

from torchgeo_bench.cka.hooks import HookCollector, _resolve_module


class _TokenModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.arange(b * 4 * 6, dtype=torch.float32).reshape(b, 4, 6)


class _SpatialModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.arange(b * 6 * 3 * 2, dtype=torch.float32).reshape(b, 6, 3, 2)


class _FlatModule(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.arange(b * 5, dtype=torch.float32).reshape(b, 5)


def test_hook_collector_vit_shape():
    model = nn.Sequential(_TokenModule())
    with HookCollector(model, ["0"]) as hc:
        model(torch.randn(3, 2))
        out = hc.collect()
    assert out["0"].shape == (3, 6)
    assert out["0"].dtype == np.float32


def test_hook_collector_cnn_shape():
    model = nn.Sequential(_SpatialModule())
    with HookCollector(model, ["0"]) as hc:
        model(torch.randn(3, 2))
        out = hc.collect()
    assert out["0"].shape == (3, 6)
    assert out["0"].dtype == np.float32


def test_hook_collector_flat_shape():
    model = nn.Sequential(_FlatModule())
    with HookCollector(model, ["0"]) as hc:
        model(torch.randn(3, 2))
        out = hc.collect()
    assert out["0"].shape == (3, 5)
    assert out["0"].dtype == np.float32


def test_hook_collector_clears_after_collect():
    model = nn.Sequential(_FlatModule())
    with HookCollector(model, ["0"]) as hc:
        model(torch.randn(3, 2))
        first = hc.collect()
        second = hc.collect()
    assert first["0"].shape == (3, 5)
    assert second["0"].shape[0] == 0


def test_hook_collector_remove_deregisters():
    model = nn.Sequential(_FlatModule())
    hc = HookCollector(model, ["0"])
    hc.remove()
    model(torch.randn(3, 2))
    out = hc.collect()
    assert out["0"].shape[0] == 0


def test_hook_collector_context_manager():
    model = nn.Sequential(_FlatModule())
    with HookCollector(model, ["0"]) as hc:
        model(torch.randn(2, 2))
        inside = hc.collect()
    model(torch.randn(2, 2))
    outside = hc.collect()
    assert inside["0"].shape == (2, 5)
    assert outside["0"].shape[0] == 0


def test_hook_collector_multi_path():
    model = nn.Module()
    model.a = _FlatModule()
    model.b = _SpatialModule()

    with HookCollector(model, ["a", "b"]) as hc:
        model.a(torch.randn(4, 2))
        model.b(torch.randn(4, 2))
        out = hc.collect()
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (4, 5)
    assert out["b"].shape == (4, 6)


def test_resolve_module_integer_index():
    model = nn.Module()
    model.blocks = nn.ModuleList([nn.Identity(), nn.ReLU()])
    resolved = _resolve_module(model, "blocks.1")
    assert isinstance(resolved, nn.ReLU)
