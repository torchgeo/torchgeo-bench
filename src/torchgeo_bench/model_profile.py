"""Measure model compute cost and prediction speed.

Measure each (model, dataset, bands) combination without dataloader overhead. Reports:

- ``throughput_samples_per_sec`` — sustained samples/s on a fixed batch
- ``latency_ms_per_batch_p50`` — median per-batch forward latency
- ``peak_gpu_mem_gb`` — peak CUDA memory during measurement
- ``params_m`` — total parameter count (millions)
- ``reserved_gpu_mem_gb`` — GPU memory held by PyTorch, including cached unused blocks
- ``gflops`` — per-sample FLOPs / 1e9 from ``torch.utils.flop_counter``
"""

import logging
import time
from dataclasses import dataclass
from statistics import median

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ProfileTiming:
    """Batch size and iteration counts for a timing measurement."""

    batch_size: int
    n_warmup: int
    n_measure: int


def _count_params(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def _count_gflops(model: nn.Module, sample: torch.Tensor | list[torch.Tensor]) -> float:
    """Count one frozen forward pass without changing the model's parameters."""
    from torch.utils.flop_counter import FlopCounterMode

    inputs = (
        sample[:1].detach()
        if isinstance(sample, torch.Tensor)
        else [feature[:1].detach() for feature in sample]
    )
    parameters = {name: parameter.detach() for name, parameter in model.named_parameters()}
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        torch.func.functional_call(model, parameters, (inputs,))
    return float(counter.get_total_flops()) / 1e9


def measure_profile(
    model: nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
    n_warmup: int = 3,
    n_measure: int = 20,
) -> dict[str, float | None]:
    """Measure forward-pass time and memory on a fixed batch.

    Args:
        model: BenchModel; ``forward`` includes normalization and ``_forward_patch_features``.
        sample_batch: representative batch shaped ``(B, C, H, W)``, already
            on ``device``.
        device: torch device used for the measurement.
        n_warmup: forward passes discarded before timing.
        n_measure: timed forward passes.

    Returns:
        Metric values, or ``None`` where unavailable (for example, GPU memory on CPU).
    """
    model.eval()
    batch_size = sample_batch.shape[0]
    is_cuda = device.type == "cuda"

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(sample_batch)

        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        per_batch_ms: list[float] = []
        t0 = time.perf_counter()
        for _ in range(n_measure):
            tb0 = time.perf_counter()
            model(sample_batch)
            if is_cuda:
                torch.cuda.synchronize(device)
            per_batch_ms.append((time.perf_counter() - tb0) * 1000.0)
        total_s = time.perf_counter() - t0

    throughput = (batch_size * n_measure) / total_s
    latency_p50 = median(per_batch_ms)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if is_cuda else None
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3 if is_cuda else None
    gflops = _count_gflops(model, sample_batch)
    params_m = _count_params(model)

    return {
        "throughput_samples_per_sec": float(throughput),
        "latency_ms_per_batch_p50": float(latency_p50),
        "peak_gpu_mem_gb": float(peak_gb) if peak_gb is not None else None,
        "reserved_gpu_mem_gb": float(reserved_gb) if reserved_gb is not None else None,
        "params_m": float(params_m),
        "gflops": gflops,
    }


def measure_cpu_throughput(
    model: nn.Module,
    sample: torch.Tensor,
    timing: ProfileTiming,
    time_budget_s: float,
) -> dict[str, float | None]:
    """Return ``*_cpu`` throughput and latency within a wall-time budget.

    Move the model to CPU for measurement, then restore its original device.
    Return ``None`` metrics with a warning if warmup exceeds ``time_budget_s``.
    Otherwise, stop after the first completed batch that exceeds the budget.
    """
    none_result: dict[str, float | None] = {
        "throughput_samples_per_sec_cpu": None,
        "latency_ms_per_batch_p50_cpu": None,
    }
    cpu_dev = torch.device("cpu")
    # Parameter-free baselines use the sample's device as their restoration target.
    first_param = next(iter(model.parameters()), None)
    orig_dev = first_param.device if first_param is not None else sample.device
    cpu_sample = sample[: timing.batch_size].detach().to(cpu_dev)
    model.to(cpu_dev)
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(timing.n_warmup):
                model(cpu_sample)
                if time.perf_counter() - t0 > time_budget_s:
                    logger.warning(
                        "[profile] CPU warmup exceeded %ss budget on %s; skipping CPU throughput.",
                        time_budget_s,
                        type(model).__name__,
                    )
                    return none_result
            per_batch_ms: list[float] = []
            t_loop = time.perf_counter()
            for _ in range(timing.n_measure):
                tb = time.perf_counter()
                model(cpu_sample)
                per_batch_ms.append((time.perf_counter() - tb) * 1000.0)
                if time.perf_counter() - t0 > time_budget_s:
                    break
            elapsed = time.perf_counter() - t_loop
        if not per_batch_ms:
            return none_result
        return {
            "throughput_samples_per_sec_cpu": (timing.batch_size * len(per_batch_ms)) / elapsed,
            "latency_ms_per_batch_p50_cpu": median(per_batch_ms),
        }
    finally:
        model.to(orig_dev)
