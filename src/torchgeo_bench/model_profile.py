"""Backbone compute/efficiency profile metrics for benchmark models.

Measured once per (model, dataset, bands) combination, isolated from
dataloader overhead. Reports:

- ``throughput_samples_per_sec`` — sustained samples/s on a fixed batch
- ``latency_ms_per_batch_p50`` — median per-batch forward latency
- ``peak_gpu_mem_gb`` — peak CUDA memory during measurement
- ``params_m`` — total parameter count (millions)
- ``reserved_gpu_mem_gb`` — allocator-reserved VRAM (vs ``peak`` which is
  the actually-used high-water mark; ratio reveals fragmentation)
- ``gflops`` — FLOPs for one sample via ``torch.utils.flop_counter``
  (stdlib, no extra dep; handles modern ops like SDPA / ViT attention)

All metrics work without extras.
"""

import logging
import time
from statistics import median

import torch
from torch import nn

logger = logging.getLogger(__name__)


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
    """Run forward-only timing + memory profile on a fixed batch.

    Args:
        model: BenchModel (its ``forward`` goes through normalization +
            ``_forward_patch_features``).
        sample_batch: representative batch shaped ``(B, C, H, W)``, already
            on ``device``.
        device: torch device used for the measurement.
        n_warmup: forward passes discarded before timing.
        n_measure: timed forward passes.

    Returns:
        Mapping of metric name to value; entries may be ``None`` when the
        underlying probe is unavailable (e.g. CPU device → no GPU-memory metrics).
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
    *,
    batch_size: int,
    n_warmup: int,
    n_measure: int,
    time_budget_s: float,
) -> dict[str, float | None]:
    """Wall-clock-budgeted CPU pass; returns ``*_cpu`` throughput and latency.

    The model and a fresh batch are moved to CPU for the duration, then moved
    back so the rest of the pipeline can keep using CUDA.  If even the first
    warmup pass exceeds ``time_budget_s`` (big ViT-L backbones can take
    minutes per batch on CPU), the metrics are returned as None with a
    warning instead of burning the budget.
    """
    none_result: dict[str, float | None] = {
        "throughput_samples_per_sec_cpu": None,
        "latency_ms_per_batch_p50_cpu": None,
    }
    cpu_dev = torch.device("cpu")
    # rcf/imagestats baselines have no parameters; use the input sample's
    # device as the restoration target since model.to() is a no-op anyway.
    first_param = next(iter(model.parameters()), None)
    orig_dev = first_param.device if first_param is not None else sample.device
    cpu_sample = sample[:batch_size].detach().to(cpu_dev)
    model.to(cpu_dev)
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_warmup):
                model(cpu_sample)
                if time.perf_counter() - t0 > time_budget_s:
                    logger.warning(
                        f"[profile] CPU warmup exceeded {time_budget_s}s budget on "
                        f"{type(model).__name__}; skipping CPU throughput."
                    )
                    return none_result
            per_batch_ms: list[float] = []
            t_loop = time.perf_counter()
            for _ in range(n_measure):
                tb = time.perf_counter()
                model(cpu_sample)
                per_batch_ms.append((time.perf_counter() - tb) * 1000.0)
                if time.perf_counter() - t0 > time_budget_s:
                    break
            elapsed = time.perf_counter() - t_loop
        if not per_batch_ms:
            return none_result
        return {
            "throughput_samples_per_sec_cpu": (batch_size * len(per_batch_ms)) / elapsed,
            "latency_ms_per_batch_p50_cpu": median(per_batch_ms),
        }
    finally:
        model.to(orig_dev)
