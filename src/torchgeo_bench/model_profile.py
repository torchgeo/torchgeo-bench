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
  (stdlib, no extra dep; handles modern ops like SDPA / ViT attention
  that fvcore misses)
- ``gpu_power_w_avg`` — mean GPU power draw during the timed loop (NVIDIA only)
- ``energy_wh_per_1k_samples`` — derived from power * time / samples
- ``sm_utilization_avg`` — mean GPU compute-utilization percentage during
  the timed loop (NVIDIA only)

Energy/power/SM-utilization require ``pynvml`` (in the ``[profile]``
extra). All other metrics work without extras.
"""

import contextlib
import logging
import threading
import time
from statistics import median

import torch
from torch import nn

logger = logging.getLogger(__name__)


def _count_params(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6


def _count_gflops(model: nn.Module, sample: torch.Tensor) -> float | None:
    """Run one forward pass under torch's FlopCounterMode for a single sample.

    Uses ``torch.utils.flop_counter`` (in stdlib torch since 2.0) — no extra
    deps and handles modern ops (SDPA, ViT attention) that fvcore misses.
    Returns ``None`` if the import is unavailable or the counter errors.
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        logger.info("torch.utils.flop_counter unavailable — GFLOPs disabled.")
        return None

    one = sample[:1]
    try:
        with FlopCounterMode(display=False) as counter, torch.inference_mode():
            model(one)
        return float(counter.get_total_flops()) / 1e9
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"FlopCounterMode failed: {exc}")
        return None


class _NvmlSampler:
    """Background thread polling NVML power (mW) and GPU utilization (%).

    No-ops when pynvml is unavailable or NVML init fails; ``samples_w``
    and ``samples_sm_util`` stay empty and callers should treat the
    derived metrics as ``None``.
    """

    def __init__(self, gpu_index: int, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples_w: list[float] = []
        self.samples_sm_util: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._pynvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception as exc:  # noqa: BLE001
            logger.info(f"pynvml unavailable — GPU power/util disabled ({exc}).")

    def __enter__(self) -> "_NvmlSampler":
        if self._handle is None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
        if self._pynvml is not None:
            with contextlib.suppress(Exception):
                self._pynvml.nvmlShutdown()

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                mw = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples_w.append(mw / 1000.0)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples_sm_util.append(float(util.gpu))
            except Exception:  # noqa: BLE001
                break
            self._stop.wait(self.interval_s)


def measure_profile(
    model: nn.Module,
    sample_batch: torch.Tensor,
    device: torch.device,
    n_warmup: int = 3,
    n_measure: int = 20,
) -> dict[str, float | None]:
    """Run forward-only timing + memory + energy profile on a fixed batch.

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
        underlying probe is unavailable (no GPU, no pynvml, no fvcore).
    """
    model.eval()
    batch_size = sample_batch.shape[0]
    is_cuda = device.type == "cuda"
    gpu_index = device.index if is_cuda and device.index is not None else 0

    with torch.inference_mode():
        for _ in range(n_warmup):
            model(sample_batch)

        if is_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        per_batch_ms: list[float] = []
        with _NvmlSampler(gpu_index) as nvml:
            t0 = time.perf_counter()
            for _ in range(n_measure):
                tb0 = time.perf_counter()
                model(sample_batch)
                if is_cuda:
                    torch.cuda.synchronize(device)
                per_batch_ms.append((time.perf_counter() - tb0) * 1000.0)
            total_s = time.perf_counter() - t0
            power_samples = list(nvml.samples_w)
            util_samples = list(nvml.samples_sm_util)

    throughput = (batch_size * n_measure) / total_s
    latency_p50 = median(per_batch_ms)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if is_cuda else None
    reserved_gb = torch.cuda.memory_reserved(device) / 1024**3 if is_cuda else None
    gflops = _count_gflops(model, sample_batch)
    params_m = _count_params(model)

    if power_samples:
        gpu_power_w_avg: float | None = sum(power_samples) / len(power_samples)
        energy_wh = gpu_power_w_avg * total_s / 3600.0
        energy_wh_per_1k: float | None = energy_wh * 1000.0 / (batch_size * n_measure)
    else:
        gpu_power_w_avg = None
        energy_wh_per_1k = None
    sm_util_avg = sum(util_samples) / len(util_samples) if util_samples else None

    return {
        "throughput_samples_per_sec": float(throughput),
        "latency_ms_per_batch_p50": float(latency_p50),
        "peak_gpu_mem_gb": float(peak_gb) if peak_gb is not None else None,
        "reserved_gpu_mem_gb": float(reserved_gb) if reserved_gb is not None else None,
        "params_m": float(params_m),
        "gflops": gflops,
        "gpu_power_w_avg": gpu_power_w_avg,
        "energy_wh_per_1k_samples": energy_wh_per_1k,
        "sm_utilization_avg": sm_util_avg,
    }
