"""Unit tests for model profiling helpers."""

import sys
import time

import pytest
import torch
from torch import nn

from torchgeo_bench.model_profile import (
    _count_gflops,
    _count_params,
    _NvmlSampler,
    measure_profile,
)


def test_count_params_correct() -> None:
    model = nn.Linear(4, 8)
    assert _count_params(model) == pytest.approx(40 / 1e6)


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


def test_measure_profile_nvml_absent_no_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _import_without_nvml(name: str, *args, **kwargs):
        if name == "pynvml":
            raise ImportError("simulated missing pynvml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_without_nvml)

    result = measure_profile(
        nn.Linear(4, 2),
        sample_batch=torch.rand(4, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=2,
    )

    assert result["gpu_power_w_avg"] is None
    assert result["energy_wh_per_1k_samples"] is None
    assert result["sm_utilization_avg"] is None


def test_count_gflops_inference_attrerror_falls_back_to_no_grad() -> None:
    class InferenceAttrErrorModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if torch.is_inference_mode_enabled():
                raise AttributeError("next_functions")
            return self.conv(x)

    gflops = _count_gflops(InferenceAttrErrorModel(), torch.rand(2, 3, 8, 8))
    assert gflops >= 0
    assert torch.isfinite(torch.tensor(gflops))


def test_count_gflops_assertion_chain_raises_not_implemented() -> None:
    class AssertionChainModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            del x
            if torch.is_inference_mode_enabled():
                raise AttributeError("next_functions")
            raise AssertionError("Expected gradient function to be set")

    with pytest.raises(NotImplementedError, match="incompatible"):
        _count_gflops(AssertionChainModel(), torch.rand(1, 3, 8, 8))


def _install_fake_pynvml(
    monkeypatch: pytest.MonkeyPatch, *, fail_on_second_power: bool = False
) -> None:
    class FakeError(Exception):
        pass

    class FakeUtil:
        def __init__(self, gpu: int) -> None:
            self.gpu = gpu

    class FakeNvml:
        NVMLError = FakeError

        def __init__(self) -> None:
            self.calls = 0

        def nvmlInit(self) -> None:
            return None

        def nvmlShutdown(self) -> None:
            return None

        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            return f"gpu{index}"

        def nvmlDeviceGetPowerUsage(self, handle: str) -> float:
            del handle
            self.calls += 1
            if fail_on_second_power and self.calls >= 2:
                raise FakeError("poll failed")
            return 50000.0

        def nvmlDeviceGetUtilizationRates(self, handle: str) -> FakeUtil:
            del handle
            return FakeUtil(gpu=42)

    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml())


def test_nvml_sampler_collects_samples_with_live_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pynvml(monkeypatch)
    sampler = _NvmlSampler(gpu_index=0, interval_s=0.001)
    with sampler:
        time.sleep(0.01)

    assert sampler.samples_w
    assert sampler.samples_sm_util
    assert sampler.samples_w[0] == pytest.approx(50.0)
    assert sampler.samples_sm_util[0] == pytest.approx(42.0)


def test_nvml_sampler_poll_stops_on_nvml_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_pynvml(monkeypatch, fail_on_second_power=True)
    sampler = _NvmlSampler(gpu_index=0, interval_s=0.001)
    sampler._poll()
    assert len(sampler.samples_w) == 1
    assert len(sampler.samples_sm_util) == 1


def test_measure_profile_uses_power_samples_for_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSampler:
        def __init__(self, gpu_index: int) -> None:
            del gpu_index
            self.samples_w = [20.0, 30.0]
            self.samples_sm_util = [70.0, 80.0]

        def __enter__(self) -> "_FakeSampler":
            return self

        def __exit__(self, *_exc) -> None:
            return None

    monkeypatch.setattr("torchgeo_bench.model_profile._NvmlSampler", _FakeSampler)
    model = nn.Linear(4, 2)
    result = measure_profile(
        model,
        sample_batch=torch.rand(4, 4),
        device=torch.device("cpu"),
        n_warmup=0,
        n_measure=2,
    )

    assert result["gpu_power_w_avg"] == pytest.approx(25.0)
    assert result["energy_wh_per_1k_samples"] is not None
    assert result["sm_utilization_avg"] == pytest.approx(75.0)
