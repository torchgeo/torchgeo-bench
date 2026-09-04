"""Measure the inference cost of a benchmark model.

The profiler deliberately measures one fixed batch. Data loading, host
transfers, and automatic batch-size searches are outside its scope.
"""

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from statistics import median

import torch
from torch import nn


@dataclass(frozen=True)
class FlopMeasurement:
    """Result of an optional FLOP measurement."""

    gflops: float | None
    status: str
    reason: str | None = None
    convention: str = "one multiply-add is two operations"
    counter: str = "torch.utils.flop_counter.FlopCounterMode"


@dataclass(frozen=True)
class ProfileResult:
    """Measurements for one explicit model and input configuration."""

    params_m: float
    throughput_samples_per_sec: float
    latency_ms_per_batch_p50: float
    peak_gpu_mem_gb: float | None
    reserved_gpu_mem_gb: float | None
    flops: FlopMeasurement
    batch_size: int
    device: str
    precision: str
    warmup: int
    measurements: int
    input_shape: tuple[int, ...]

    def as_dict(self) -> dict[str, float | int | str | None]:
        """Return a flat representation suitable for result storage."""
        return {
            "params_m": self.params_m,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
            "latency_ms_per_batch_p50": self.latency_ms_per_batch_p50,
            "peak_gpu_mem_gb": self.peak_gpu_mem_gb,
            "reserved_gpu_mem_gb": self.reserved_gpu_mem_gb,
            "gflops": self.flops.gflops,
            "gflops_status": self.flops.status,
            "gflops_reason": self.flops.reason,
            "gflops_counter": self.flops.counter,
            "gflops_convention": self.flops.convention,
            "batch_size": self.batch_size,
            "device": self.device,
            "precision": self.precision,
            "warmup": self.warmup,
            "measurements": self.measurements,
            "input_shape": str(self.input_shape),
        }


def _count_params(model: nn.Module) -> float:
    """Return the number of model parameters in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def _count_gflops(model: nn.Module, sample: torch.Tensor) -> float:
    """Count FLOPs for one sample with PyTorch's standard counter."""
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as counter, torch.inference_mode():
        model(sample[:1])
    return float(counter.get_total_flops()) / 1e9


def _precision_context(
    device: torch.device, precision: str
) -> contextlib.AbstractContextManager[None]:
    if precision == "float32":
        return contextlib.nullcontext()
    if precision == "float16":
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    if precision == "bfloat16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError("precision must be 'float32', 'float16', or 'bfloat16'")


@contextlib.contextmanager
def _evaluation_mode(model: nn.Module) -> Iterator[None]:
    states = {module: module.training for module in model.modules()}
    model.eval()
    try:
        yield
    finally:
        for module, training in states.items():
            module.train(training)


def profile_inference(
    model: nn.Module,
    sample_batch: torch.Tensor,
    *,
    device: torch.device,
    precision: str = "float32",
    n_warmup: int = 3,
    n_measure: int = 20,
    count_flops: bool = True,
) -> ProfileResult:
    """Measure one fixed inference configuration.

    Args:
        model: Model to evaluate.
        sample_batch: Input batch already placed on *device*.
        device: Device used for inference.
        precision: Inference precision policy.
        n_warmup: Number of untimed warmup passes.
        n_measure: Number of timed passes.
        count_flops: Whether to run the optional FLOP counter.

    Returns:
        Parameters, timing, memory, and FLOP measurements.

    Raises:
        ValueError: If iteration counts, batch shape, device, or precision is
            invalid.
    """
    if n_warmup < 0 or n_measure <= 0:
        raise ValueError("n_warmup must be non-negative and n_measure must be positive")
    if sample_batch.ndim == 0 or sample_batch.shape[0] <= 0:
        raise ValueError("sample_batch must have a non-empty batch dimension")
    if sample_batch.device != device:
        raise ValueError(f"sample_batch is on {sample_batch.device}, expected {device}")
    if precision not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be 'float32', 'float16', or 'bfloat16'")
    batch_size = sample_batch.shape[0]
    is_cuda = device.type == "cuda"
    with _evaluation_mode(model), torch.inference_mode(), _precision_context(device, precision):
        for _ in range(n_warmup):
            model(sample_batch)
        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        started = time.perf_counter()
        for _ in range(n_measure):
            pass_started = time.perf_counter()
            model(sample_batch)
            if is_cuda:
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - pass_started) * 1000)
        elapsed = time.perf_counter() - started
    if count_flops:
        flops = FlopMeasurement(_count_gflops(model, sample_batch), "measured")
    else:
        flops = FlopMeasurement(None, "disabled", "FLOP counting was not requested")
    return ProfileResult(
        params_m=_count_params(model),
        throughput_samples_per_sec=batch_size * n_measure / elapsed,
        latency_ms_per_batch_p50=float(median(timings)),
        peak_gpu_mem_gb=(torch.cuda.max_memory_allocated(device) / 1024**3 if is_cuda else None),
        reserved_gpu_mem_gb=(torch.cuda.memory_reserved(device) / 1024**3 if is_cuda else None),
        flops=flops,
        batch_size=batch_size,
        device=str(device),
        precision=precision,
        warmup=n_warmup,
        measurements=n_measure,
        input_shape=tuple(sample_batch.shape),
    )


def measure_profile(
    model: nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
    n_warmup: int = 3,
    n_measure: int = 20,
) -> dict[str, float | None]:
    """Compatibility wrapper returning the historical metric mapping."""
    result = profile_inference(
        model, sample_batch, device=device, n_warmup=n_warmup, n_measure=n_measure
    )
    return {
        "throughput_samples_per_sec": result.throughput_samples_per_sec,
        "latency_ms_per_batch_p50": result.latency_ms_per_batch_p50,
        "peak_gpu_mem_gb": result.peak_gpu_mem_gb,
        "reserved_gpu_mem_gb": result.reserved_gpu_mem_gb,
        "params_m": result.params_m,
        "gflops": result.flops.gflops,
    }


def measure_cpu_throughput(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    batch_size: int,
    n_warmup: int,
    n_measure: int,
    time_budget_s: float,
) -> dict[str, float | None]:
    """Measure CPU throughput in a separate, fixed-size invocation."""
    if batch_size <= 0 or batch_size > sample.shape[0]:
        raise ValueError("batch_size must be within the sample batch")
    if n_warmup < 0 or n_measure <= 0 or time_budget_s <= 0:
        raise ValueError("CPU profiling settings must be positive")
    original_device = next(model.parameters(), sample).device
    cpu_sample = sample[:batch_size].detach().to("cpu")
    states = {module: module.training for module in model.modules()}
    model.to("cpu").eval()
    try:
        with torch.inference_mode():
            for _ in range(n_warmup):
                model(cpu_sample)
            timings: list[float] = []
            started = time.perf_counter()
            for _ in range(n_measure):
                pass_started = time.perf_counter()
                model(cpu_sample)
                timings.append((time.perf_counter() - pass_started) * 1000)
                if time.perf_counter() - started >= time_budget_s:
                    break
            elapsed = time.perf_counter() - started
        return {
            "throughput_samples_per_sec_cpu": batch_size * n_measure / elapsed,
            "latency_ms_per_batch_p50_cpu": float(median(timings)),
        }
    finally:
        model.to(original_device)
        for module, training in states.items():
            module.train(training)
