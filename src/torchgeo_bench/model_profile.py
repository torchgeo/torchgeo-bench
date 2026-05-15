"""Backbone compute/efficiency profile metrics for benchmark models.

Measured once per (model, dataset, bands) combination, isolated from
dataloader overhead. Reports:

- ``throughput_samples_per_sec`` — sustained samples/s on a fixed batch
- ``latency_ms_per_batch_p50`` — median per-batch forward latency
- ``peak_gpu_mem_gb`` — peak CUDA memory during measurement
- ``params_m`` — total parameter count (millions)
- ``gmacs`` — MACs for one sample, via ``fvcore`` if installed; ``None`` otherwise
- ``gpu_power_w_avg`` — mean GPU power draw during the timed loop (NVIDIA only)
- ``energy_wh_per_1k_samples`` — derived from power * time / samples

GMACs requires ``fvcore``; energy/power requires ``pynvml`` (both in the
``[profile]`` extra). All other metrics work without extra deps.
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


class _NvmlPowerSampler:
    """Background thread that polls NVML power draw (mW) into a list.

    No-ops when pynvml is unavailable or NVML init fails; ``samples_w``
    stays empty and callers should treat power as ``None``.
    """

    def __init__(self, gpu_index: int, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.samples_w: list[float] = []
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
            logger.info(f"pynvml unavailable — GPU power/energy disabled ({exc}).")

    def __enter__(self) -> "_NvmlPowerSampler":
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
        with _NvmlPowerSampler(gpu_index) as power:
            t0 = time.perf_counter()
            for _ in range(n_measure):
                tb0 = time.perf_counter()
                model(sample_batch)
                if is_cuda:
                    torch.cuda.synchronize(device)
                per_batch_ms.append((time.perf_counter() - tb0) * 1000.0)
            total_s = time.perf_counter() - t0
            power_samples = list(power.samples_w)

    throughput = (batch_size * n_measure) / total_s
    latency_p50 = median(per_batch_ms)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if is_cuda else None
    gmacs = _count_gmacs(model, sample_batch)
    params_m = _count_params(model)

    if power_samples:
        gpu_power_w_avg: float | None = sum(power_samples) / len(power_samples)
        energy_wh = gpu_power_w_avg * total_s / 3600.0
        energy_wh_per_1k: float | None = energy_wh * 1000.0 / (batch_size * n_measure)
    else:
        gpu_power_w_avg = None
        energy_wh_per_1k = None

    return {
        "throughput_samples_per_sec": float(throughput),
        "latency_ms_per_batch_p50": float(latency_p50),
        "peak_gpu_mem_gb": float(peak_gb) if peak_gb is not None else None,
        "params_m": float(params_m),
        "gmacs": gmacs,
        "gpu_power_w_avg": gpu_power_w_avg,
        "energy_wh_per_1k_samples": energy_wh_per_1k,
    }
