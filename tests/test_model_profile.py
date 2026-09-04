"""Unit tests for model profiling helpers."""

import pytest
import torch
from torch import nn

from torchgeo_bench import model_profile
from torchgeo_bench.model_profile import (
    _count_gflops,
    _count_params,
    measure_cpu_throughput,
    measure_profile,
    profile_inference,
)


def test_count_params_correct() -> None:
    model = nn.Linear(4, 8)
    assert _count_params(model) == pytest.approx(40 / 1e6)


def test_count_gflops_uses_two_operations_per_multiply_add() -> None:
    model = nn.Linear(2, 2, bias=False)
    assert _count_gflops(model, torch.rand(1, 2)) == pytest.approx(8 / 1e9)


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
    assert result["params_m"] is not None
    assert result["params_m"] > 0


def test_measure_profile_does_not_hide_counter_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def unsupported_counter(model: nn.Module, sample: torch.Tensor) -> float:
        raise NotImplementedError("unsupported counter operation")

    monkeypatch.setattr(model_profile, "_count_gflops", unsupported_counter)
    with pytest.raises(NotImplementedError, match="unsupported counter operation"):
        measure_profile(nn.Linear(4, 2), torch.rand(2, 4), torch.device("cpu"), 0, 1)


def test_count_gflops_propagates_inference_error() -> None:
    class InferenceAttrErrorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if torch.is_inference_mode_enabled():
                raise AttributeError("next_functions")
            return self.conv(x)

    with pytest.raises(AttributeError, match="next_functions"):
        _count_gflops(InferenceAttrErrorModel(), torch.rand(2, 3, 8, 8))


def test_count_gflops_propagates_execution_error() -> None:
    class AssertionChainModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            del x
            raise RuntimeError("model execution failed")

    with pytest.raises(RuntimeError, match="model execution failed"):
        _count_gflops(AssertionChainModel(), torch.rand(1, 3, 8, 8))


def test_profile_records_fixed_batch_and_precision() -> None:
    result = profile_inference(
        nn.Linear(4, 2),
        torch.rand(3, 4),
        device=torch.device("cpu"),
        precision="float32",
        n_warmup=1,
        n_measure=2,
        count_flops=False,
    )
    assert result.batch_size == 3
    assert result.precision == "float32"
    assert result.input_shape == (3, 4)
    assert result.flops.status == "disabled"
    assert result.flops.gflops is None


def test_profile_rejects_batch_on_wrong_device() -> None:
    with pytest.raises(ValueError, match="device must be"):
        profile_inference(nn.Linear(4, 2), torch.rand(2, 4), device=torch.device("meta"))


def test_profile_cpu_reports_timing_and_memory_semantics() -> None:
    result = profile_inference(
        nn.Linear(4, 2),
        torch.rand(2, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=2,
        count_flops=False,
    )
    assert result.throughput_samples_per_sec > 0
    assert result.latency_ms_per_batch_p50 > 0
    assert result.peak_gpu_mem_gb is None
    assert result.reserved_gpu_mem_gb is None


def test_profile_propagates_unsupported_model_flops() -> None:
    class Unsupported(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls > 1:
                raise NotImplementedError("operator is unsupported")
            return x

    with pytest.raises(NotImplementedError, match="unsupported"):
        profile_inference(
            Unsupported(),
            torch.rand(1, 2),
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=1,
            count_flops=True,
        )


def test_profile_propagates_real_flop_execution_failure() -> None:
    class Broken(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("real model failure")

    with pytest.raises(RuntimeError, match="real model failure"):
        profile_inference(
            Broken(),
            torch.rand(1, 2),
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=1,
            count_flops=True,
        )


def test_profile_does_not_mutate_autograd_hooks() -> None:
    import torch.autograd.graph as autograd_graph
    import torch.utils.module_tracker as module_tracker

    graph_hook = autograd_graph.register_multi_grad_hook
    tracker_hook = module_tracker.register_multi_grad_hook
    profile_inference(
        nn.Linear(4, 2),
        torch.rand(1, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=1,
        count_flops=False,
    )
    assert autograd_graph.register_multi_grad_hook is graph_hook
    assert module_tracker.register_multi_grad_hook is tracker_hook


def test_profile_restores_mixed_training_modes() -> None:
    model = nn.Sequential(nn.Linear(4, 2), nn.BatchNorm1d(2))
    model.train()
    model[1].eval()
    profile_inference(
        model,
        torch.rand(2, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=1,
        count_flops=False,
    )
    assert model.training is True
    assert model[1].training is False


@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_profile_accepts_autocast_precisions(precision: str) -> None:
    result = profile_inference(
        nn.Linear(4, 2),
        torch.rand(1, 4),
        device=torch.device("cpu"),
        precision=precision,
        n_warmup=0,
        n_measure=1,
        count_flops=False,
    )
    assert result.precision == precision


def test_profile_rejects_invalid_settings() -> None:
    model = nn.Linear(2, 2)
    sample = torch.rand(1, 2)
    with pytest.raises(ValueError, match="n_warmup"):
        profile_inference(model, sample, device=torch.device("cpu"), n_warmup=-1)
    with pytest.raises(ValueError, match="n_measure"):
        profile_inference(model, sample, device=torch.device("cpu"), n_measure=0)
    with pytest.raises(ValueError, match="precision"):
        profile_inference(model, sample, device=torch.device("cpu"), precision="int8")


def test_profile_as_dict_includes_flop_metadata() -> None:
    result = profile_inference(
        nn.Linear(2, 2),
        torch.rand(1, 2),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=1,
        count_flops=False,
    )
    values = result.as_dict()
    assert values["gflops_status"] == "disabled"
    assert values["gflops_convention"] == "one multiply-add is two operations"
    assert values["gflops_coverage"] == "registered operators only; total coverage unverified"


def test_cpu_profile_restores_model_and_reports_metrics() -> None:
    model = nn.Linear(2, 2)
    metrics = measure_cpu_throughput(
        model,
        torch.rand(2, 2),
        batch_size=2,
        n_warmup=0,
        n_measure=1,
        time_budget_s=1,
    )
    assert metrics["throughput_samples_per_sec_cpu"] is not None
    assert metrics["latency_ms_per_batch_p50_cpu"] is not None
    assert next(model.parameters()).device == torch.device("cpu")
