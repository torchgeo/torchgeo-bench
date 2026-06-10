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

Energy/power/SM-utilization require ``nvidia-ml-py`` (in the ``[profile]``
extra). All other metrics work without extras.
"""

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


def _count_gflops(model: nn.Module, sample: torch.Tensor) -> float:
    """Run one forward pass under torch's FlopCounterMode for a single sample.

    Uses ``torch.utils.flop_counter`` (stdlib torch since 2.0).  The
    counter's ``module_tracker`` wants to call
    ``register_multi_grad_hook`` on the activations, which requires every
    intermediate tensor to have a ``grad_fn`` (i.e. autograd must be
    enabled).  Three different contexts have to be tried in order from
    cheapest to most permissive:

    1. ``torch.inference_mode``: cheapest.  Works for most timm /
       torchgeo backbones because their forwards don't trigger the
       module-tracker code paths that need grad_fn.
    2. ``torch.no_grad``: keeps the version counter that
       ``inference_mode`` disables.  Catches forwards that rely on
       ``inference_tensor`` returning a regular tensor.
    3. ``torch.enable_grad`` with ``requires_grad=True`` on the input:
       only path that gives every activation a grad_fn.  Required by
       ``torchgeo.models.Panopticon``, which inside ``MultiheadAttention``
       triggers ``register_multi_grad_hook`` on a non-leaf tensor and
       raises ``AssertionError: Expected gradient function to be set``
       otherwise.  More expensive (autograd graph is built and then
       discarded) but still a single forward pass.

    Each fallback only triggers on the *specific* exception that earlier
    fallback can't handle, so genuine bugs in the counter surface
    instead of being papered over.
    """
    from torch.utils.flop_counter import FlopCounterMode

    base = sample[:1]

    def _measure_no_autograd(ctx_factory) -> float:
        with FlopCounterMode(display=False) as counter, ctx_factory():
            model(base)
        return float(counter.get_total_flops()) / 1e9

    def _measure_with_autograd() -> float:
        inp = base.detach().clone().requires_grad_(True)
        with FlopCounterMode(display=False) as counter, torch.enable_grad():
            model(inp)
        return float(counter.get_total_flops()) / 1e9

    try:
        return _measure_no_autograd(torch.inference_mode)
    except AttributeError as exc:
        # 'NoneType' object has no attribute 'next_functions' — only
        # the dispatch-vs-inference-tensor mismatch.
        if "next_functions" not in str(exc):
            raise
        logger.info(
            f"FlopCounterMode inference_mode rejected by {type(model).__name__}: "
            f"{exc}; retrying under no_grad."
        )
    try:
        return _measure_no_autograd(torch.no_grad)
    except AssertionError as exc:
        # 'Expected gradient function to be set' — the module_tracker
        # needs activations to have grad_fn; only enable_grad provides that.
        if "Expected gradient function" not in str(exc):
            raise
        logger.info(
            f"FlopCounterMode no_grad rejected by {type(model).__name__}: "
            f"{exc}; retrying under enable_grad with requires_grad input."
        )
    try:
        return _measure_with_autograd()
    except AssertionError as exc:
        # 'Expected gradient function to be set' even after enable_grad +
        # requires_grad=True on the input.  Observed on
        # torchgeo.models.Panopticon: its PanopticonChnFusion path
        # introduces a tensor (chn_ids / mask / conv3d input) that breaks
        # the autograd chain before module_tracker's pre-hook fires.
        # There are no more contexts to try from this module; a fix has
        # to land in panopticon.py upstream (carry requires_grad through
        # chnfus) or in torch's module_tracker (don't require grad_fn).
        # Raise a typed signal so the caller can record gflops=None *for
        # this specific known-incompatible model* without a generic
        # swallow.
        if "Expected gradient function" not in str(exc):
            raise
        raise NotImplementedError(
            f"{type(model).__name__} is incompatible with "
            f"torch.utils.flop_counter.FlopCounterMode: all three execution "
            f"contexts (inference_mode, no_grad, enable_grad+requires_grad) "
            f"fail with the same AssertionError from module_tracker — its "
            f"forward graph drops grad_fn before the pre-hook fires.  GFLOPs "
            f"cannot be measured for this model with the current tooling; "
            f"fix needs to land in the model's own forward or in torch."
        ) from exc


class _NvmlSampler:
    """Background thread polling NVML power (mW) and GPU utilization (%).

    No-ops when nvidia-ml-py is unavailable or NVML init fails; ``samples_w``
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
        except ImportError:
            logger.info("nvidia-ml-py not installed — GPU power/util disabled.")
            return
        try:
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except pynvml.NVMLError as exc:
            logger.info(
                f"nvidia-ml-py installed but NVML init failed — GPU power/util disabled ({exc})."
            )

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
            try:
                self._pynvml.nvmlShutdown()
            except self._pynvml.NVMLError as exc:
                logger.warning(f"nvmlShutdown failed: {exc} (likely benign at exit)")

    def _poll(self) -> None:
        nvml = self._pynvml
        while not self._stop.is_set():
            try:
                mw = nvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples_w.append(mw / 1000.0)
                util = nvml.nvmlDeviceGetUtilizationRates(self._handle)
                self.samples_sm_util.append(float(util.gpu))
            except nvml.NVMLError as exc:
                # Specific NVML error mid-run is worth knowing about; don't
                # silently degrade.  Log and stop polling but keep what
                # samples we have.
                logger.warning(
                    f"NVML poll failed after {len(self.samples_w)} samples ({exc}); "
                    f"stopping power/util sampling for this measurement."
                )
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
        underlying probe is unavailable (no GPU, no nvidia-ml-py, no fvcore).
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
    # Catch only the typed signal _count_gflops raises for known-
    # incompatible models (currently Panopticon).  Any other exception
    # propagates so we keep investigating new failure modes instead of
    # silently writing None.
    try:
        gflops: float | None = _count_gflops(model, sample_batch)
    except NotImplementedError as exc:
        logger.warning(f"[profile] {exc}")
        gflops = None
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
