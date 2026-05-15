"""Backbone-only performance metrics for benchmark models.

Measured once per (model, dataset, bands) combination, isolated from
dataloader overhead. Reports:

- ``throughput_samples_per_sec`` — sustained samples/s on a fixed batch
- ``latency_ms_per_batch_p50`` — median per-batch forward latency
- ``peak_gpu_mem_gb`` — peak CUDA memory during measurement
- ``params_m`` — total parameter count (millions)
- ``gmacs`` — multiply-accumulate count for one sample, via ``fvcore`` if
  installed; ``None`` otherwise

GMACs depend on ``fvcore`` (in the ``[perf]`` extra). All other metrics
work without extra deps.
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


def _count_gmacs(model: nn.Module, sample: torch.Tensor) -> float | None:
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        logger.info("fvcore not installed — GMACs unavailable (`pip install fvcore`).")
        return None

    one = sample[:1]
    try:
        flop = FlopCountAnalysis(model, one)
        flop.unsupported_ops_warnings(False)
        flop.uncalled_modules_warnings(False)
        return float(flop.total()) / 1e9
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"fvcore FlopCountAnalysis failed: {exc}")
        return None


def measure_model_perf(
    model: nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
    n_warmup: int = 3,
    n_measure: int = 20,
) -> dict[str, float | None]:
    """Run forward-only timing + memory profiling on a fixed batch.

    Args:
        model: BenchModel (its ``forward`` goes through normalization +
            ``_forward_patch_features``).
        sample_batch: representative batch shaped ``(B, C, H, W)``, already
            on ``device``.
        device: torch device used for the measurement.
        n_warmup: forward passes discarded before timing.
        n_measure: timed forward passes.

    Returns:
        Mapping with keys ``throughput_samples_per_sec``,
        ``latency_ms_per_batch_p50``, ``peak_gpu_mem_gb``, ``params_m``,
        ``gmacs`` (GMACs may be ``None``).
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
    gmacs = _count_gmacs(model, sample_batch)
    params_m = _count_params(model)

    return {
        "throughput_samples_per_sec": float(throughput),
        "latency_ms_per_batch_p50": float(latency_p50),
        "peak_gpu_mem_gb": float(peak_gb) if peak_gb is not None else None,
        "params_m": float(params_m),
        "gmacs": gmacs,
    }
