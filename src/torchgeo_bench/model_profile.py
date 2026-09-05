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

import contextlib
import logging
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Literal

import torch
import torch.autograd.graph as autograd_graph
import torch.utils.module_tracker as module_tracker
from torch import nn
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ProfileTiming:
    """Batch size and iteration counts for a timing measurement."""

    batch_size: int
    n_warmup: int
    n_measure: int


class FlopCountUnavailableError(NotImplementedError):
    """The FLOP counter cannot track this model's gradient functions."""


@contextlib.contextmanager
def lenient_grad_hooks() -> Iterator[None]:
    """Let ``module_tracker`` skip its bookkeeping for grad-less tensors.

    ``FlopCounterMode`` counts at the ``TorchDispatchMode`` layer, which
    never needs ``grad_fn``; only ``module_tracker``'s *per-module
    attribution* does.  Its forward-pre-hook calls
    ``register_multi_grad_hook`` on the module inputs, which asserts that
    every input tensor has a ``grad_fn``.  Models that break the autograd
    chain inside their forward — ``torchgeo.models.Panopticon`` does, via
    ``self.conv3d(x.unsqueeze(1))`` in its patch-embed — trip that
    assertion *before any counting happens*, so all three execution
    contexts in :func:`_count_gflops` fail identically.

    Swallowing the assertion loses module-level breakdown for those
    modules, not FLOPs: the dispatch-layer totals are unaffected.

    ``module_tracker`` binds the symbol at import time
    (``from torch.autograd.graph import register_multi_grad_hook``), so
    both the source module and that binding have to be patched.
    """
    original = autograd_graph.register_multi_grad_hook

    def safe(
        tensors: Sequence[torch.Tensor],
        fn: Callable[[Sequence[torch.Tensor | None]], None] | Callable[[torch.Tensor], None],
        *,
        mode: Literal["all", "any"] = "all",
    ) -> RemovableHandle:
        try:
            return original(tensors, fn, mode=mode)
        except AssertionError as exc:  # allow-except: Skip grad_fn bookkeeping, retaining FLOPs.
            if "Expected gradient function" not in str(exc):
                raise
            return RemovableHandle(OrderedDict())

    autograd_graph.register_multi_grad_hook = safe
    module_tracker.register_multi_grad_hook = safe
    try:
        yield
    finally:
        autograd_graph.register_multi_grad_hook = original
        module_tracker.register_multi_grad_hook = original


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

    The whole ladder runs inside :func:`lenient_grad_hooks`, which makes
    ``module_tracker``'s grad_fn assertion non-fatal.  Models that used to
    exhaust all three tiers (Panopticon) now resolve early; models that
    already worked are unaffected bit-for-bit, since the patch only
    intercepts an error path they never take.
    """
    from torch.utils.flop_counter import FlopCounterMode

    base = sample[:1]

    def _measure_no_autograd(ctx_factory) -> float:
        with lenient_grad_hooks(), FlopCounterMode(display=False) as counter, ctx_factory():
            model(base)
        return float(counter.get_total_flops()) / 1e9

    def _measure_with_autograd() -> float:
        inp = base.detach().clone().requires_grad_(True)
        with lenient_grad_hooks(), FlopCounterMode(display=False) as counter, torch.enable_grad():
            model(inp)
        return float(counter.get_total_flops()) / 1e9

    try:
        return _measure_no_autograd(torch.inference_mode)
    except AttributeError as exc:  # allow-except: Retry FLOP counting without inference tensors.
        # 'NoneType' object has no attribute 'next_functions' — only
        # the dispatch-vs-inference-tensor mismatch.
        if "next_functions" not in str(exc):
            raise
        logger.info(
            "FlopCounterMode inference_mode rejected by %s: %s; retrying under no_grad.",
            type(model).__name__,
            exc,
        )
    try:
        return _measure_no_autograd(torch.no_grad)
    except AssertionError as exc:  # allow-except: Handle the FLOP counter grad_fn assertion.
        # 'Expected gradient function to be set' — the module_tracker
        # needs activations to have grad_fn; only enable_grad provides that.
        if "Expected gradient function" not in str(exc):
            raise
        logger.info(
            "FlopCounterMode no_grad rejected by %s: %s; retrying with requires_grad input.",
            type(model).__name__,
            exc,
        )
    try:
        return _measure_with_autograd()
    except AssertionError as exc:  # allow-except: Handle the FLOP counter grad_fn assertion.
        # The lenient patch neutralises every module_tracker assertion we
        # know of, so reaching here means the assertion no longer comes from
        # module_tracker; raise a typed signal the caller can record as
        # gflops=None.
        if "Expected gradient function" not in str(exc):
            raise
        raise FlopCountUnavailableError(
            f"{type(model).__name__} is incompatible with "
            f"torch.utils.flop_counter.FlopCounterMode: all three execution "
            f"contexts (inference_mode, no_grad, enable_grad+requires_grad) "
            f"fail with an AssertionError that lenient_grad_hooks() does not "
            f"intercept, so it does not originate from module_tracker's "
            f"grad_fn bookkeeping.  GFLOPs cannot be measured for this model "
            f"with the current tooling."
        ) from exc


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
        underlying probe is unavailable (e.g. CPU device → no GPU-memory
        metrics, or a model the FLOP counter can't trace → no ``gflops``).
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
    try:
        gflops: float | None = _count_gflops(model, sample_batch)
    except FlopCountUnavailableError as exc:  # allow-except: Timing remains valid without FLOPs.
        logger.warning("[profile] %s", exc)
        gflops = None
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
