"""Unit tests for model profiling helpers."""

import pytest
import torch
from torch import nn

from torchgeo_bench.model_profile import (
    _count_gflops,
    _count_params,
    measure_profile,
)


def test_count_params_correct() -> None:
    model = nn.Linear(4, 8)
    assert _count_params(model) == pytest.approx(40 / 1e6)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_count_gflops_matches_conv_and_linear_ops(requires_grad) -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    sample = torch.rand(1, 3, 16, 16, requires_grad=requires_grad)
    gflops = _count_gflops(model, sample)
    expected = 2 * 14 * 14 * 4 * 3 * 3 * 3 + 2 * 4 * 2
    assert gflops == pytest.approx(expected / 1e9)
    assert sample.grad is None
    assert all(p.grad is None for p in model.parameters())


def test_count_gflops_handles_parameter_views_without_mutating_model() -> None:
    class LearnedQuery(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = nn.Parameter(torch.ones(1, 4))
            self.project = nn.Linear(4, 2)
            self.project.weight.requires_grad_(False)

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return self.project(self.query.expand(images.shape[0], -1))

    model = LearnedQuery().eval()
    original_query = model.query
    assert _count_gflops(model, torch.randn(2, 3, 8, 8)) == 16 / 1e9
    assert model.query is original_query
    assert model.query.requires_grad
    assert not model.project.weight.requires_grad


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("forward not implemented"),
        AssertionError("Expected gradient function to be set"),
        AttributeError("next_functions"),
        RuntimeError("model failed"),
    ],
)
def test_count_gflops_propagates_errors_without_retry(error) -> None:
    calls = []

    class BrokenForward(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            calls.append(x)
            raise error

    with pytest.raises(type(error)) as exc:
        _count_gflops(BrokenForward(), torch.rand(1, 3, 16, 16))
    assert exc.value is error
    assert len(calls) == 1


def test_measure_profile_cpu_returns_dict() -> None:
    model = nn.Linear(4, 2)
    sample_batch = torch.rand(4, 4)
    result = measure_profile(
        model,
        sample_batch=sample_batch,
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=2,
    )

    assert isinstance(result, dict)
    assert "params_m" in result
    assert "throughput_samples_per_sec" in result
    assert result["params_m"] is not None and result["params_m"] > 0


def test_measure_profile_does_not_hide_counter_errors(monkeypatch) -> None:
    def unsupported_counter(model, sample):
        raise NotImplementedError("unsupported counter operation")

    monkeypatch.setattr("torchgeo_bench.model_profile._count_gflops", unsupported_counter)
    with pytest.raises(NotImplementedError, match="unsupported counter operation"):
        measure_profile(nn.Linear(4, 2), torch.rand(2, 4), torch.device("cpu"), 0, 1)
