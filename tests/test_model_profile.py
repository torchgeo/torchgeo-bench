"""Unit tests for model profiling helpers."""

from types import SimpleNamespace
from typing import Any, cast

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


def test_count_gflops_returns_finite() -> None:
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    gflops = _count_gflops(model, torch.rand(1, 3, 16, 16))
    assert gflops > 0
    assert torch.isfinite(torch.tensor(gflops))


def test_count_gflops_not_implemented_propagates() -> None:
    class NotImplementedForward(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            raise NotImplementedError("forward not implemented")

    with pytest.raises(NotImplementedError, match="forward not implemented"):
        _count_gflops(NotImplementedForward(), torch.rand(1, 3, 16, 16))


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
    assert result["gflops"] == pytest.approx(16 / 1e9)


def test_profile_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device("cuda"))


def test_profile_validates_tensor_devices_and_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(ValueError, match="sample_batch is on"):
        profile_inference(nn.Identity(), torch.ones(1, 2), device=torch.device("cuda:0"))
    mixed = nn.Linear(2, 2)
    mixed.register_buffer("other", torch.ones(1, device="meta"))
    with pytest.raises(ValueError, match="one device"):
        profile_inference(mixed, torch.ones(1, 2), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="model tensors are on"):
        profile_inference(
            nn.Linear(2, 2, device="meta"), torch.ones(1, 2), device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="float32"):
        profile_inference(
            nn.Linear(2, 2).double(), torch.ones(1, 2).double(), device=torch.device("cpu")
        )
    with pytest.raises(ValueError, match="precision"):
        model_profile._precision_context(torch.device("cpu"), "invalid")


def test_mocked_cuda_timing_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Identity(nn.Module):
        def forward(self, sample: Any) -> Any:
            events.append("forward")
            return sample

    def synchronize(device: torch.device) -> None:
        assert device == torch.device("cuda:0")
        events.append("sync")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda device: events.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 2 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 3 * 1024**3)
    sample = cast(
        torch.Tensor,
        SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.float32, shape=(2, 3), ndim=2),
    )
    result = profile_inference(
        Identity(), sample, device=torch.device("cuda"), n_warmup=1, n_measure=1
    )
    assert events == ["forward", "sync", "reset", "forward", "sync"]
    assert result.peak_gpu_mem_gb == 2
    assert result.reserved_gpu_mem_gb == 3
    assert result.device == "cuda:0"


def test_cpu_budget_and_denominator(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([0.0, 0.2, 0.2, 0.7, 0.7, 0.7])
    monkeypatch.setattr(model_profile.time, "perf_counter", lambda: next(timestamps))
    result = measure_cpu_throughput(
        nn.Identity(), torch.ones(2, 2), batch_size=2, n_warmup=0, n_measure=10, time_budget_s=0.5
    )
    assert result["throughput_samples_per_sec_cpu"] == pytest.approx(4)
    assert result["latency_ms_per_batch_p50_cpu"] == pytest.approx(500)


def test_cpu_budget_can_expire_during_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    timestamps = iter([0.0, 2.0])
    monkeypatch.setattr(model_profile.time, "perf_counter", lambda: next(timestamps))
    model = nn.Identity()
    result = measure_cpu_throughput(
        model, torch.ones(2, 2), batch_size=2, n_warmup=1, n_measure=1, time_budget_s=1
    )
    assert all(value is None for value in result.values())
    assert model.training


def test_cpu_invalid_settings() -> None:
    with pytest.raises(ValueError, match="non-empty batch"):
        profile_inference(nn.Identity(), torch.empty(0, 2), device=torch.device("cpu"))
    with pytest.raises(ValueError, match="sample batch"):
        measure_cpu_throughput(
            nn.Identity(), torch.ones(1, 2), batch_size=2, n_warmup=0, n_measure=1, time_budget_s=1
        )
    with pytest.raises(ValueError, match="positive"):
        measure_cpu_throughput(
            nn.Identity(), torch.ones(1, 2), batch_size=1, n_warmup=0, n_measure=0, time_budget_s=1
        )


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
