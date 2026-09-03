"""Per-sample compute cost (GFLOPs) split into backbone / head / probe.

Measures one model config across two fixed band configurations and, where
the model declares segmentation layers, across the segmentation head types.
Everything is measured on a **synthetic** tensor: ``_count_gflops`` slices
``sample[:1]``, so no dataset is involved and no data is downloaded.  Band
specs come off the dataset *class attributes* of ``cloudsen12``.

The pipeline **imports** the real eval wiring (``build_seg_probe_and_solver``,
``measure_profile``, ``_count_gflops``) rather than reimplementing it, so the
graph that gets measured is the graph the eval runs.

One invocation handles one model config, matching ``seg_corruption_pipeline``;
``slurm/eval_flops.sbatch`` loops the model set.
"""

import logging
import os
import warnings
from datetime import UTC

import torch
from torch import nn

from torchgeo_bench.config import instantiate
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.main import resolve_configured_device
from torchgeo_bench.model_profile import (
    _count_gflops,
    _count_params,
    measure_profile,
)
from torchgeo_bench.models._band_mapping import BandIncompatibilityError
from torchgeo_bench.results import append_rows_atomic
from torchgeo_bench.segmentation_task import build_seg_probe_and_solver
from torchgeo_bench.settings import FlopsSettings, merge

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)


def _load_completed(path: str) -> frozenset[tuple]:
    """Return the ``(name, band_config, task, head_type)`` keys already in *path*.

    Only the specific, well-understood ways a result file can be legitimately
    "nothing measured yet" are tolerated: empty (``EmptyDataError``),
    unparseable CSV syntax (``ParserError``), or missing one of the expected
    resume-key columns (``KeyError``, e.g. an older schema). Anything else --
    a permissions error, a truncated/binary file, an unrelated bug -- is a
    real failure and must propagate rather than be silently treated as an
    empty, freshly-resumable file.
    """
    if not os.path.exists(path):
        return frozenset()

    import pandas as pd

    try:
        df = pd.read_csv(path)
        return frozenset(
            zip(
                df["name"],
                df["band_config"],
                df["task"],
                df["head_type"].fillna(""),
                strict=False,
            )
        )
    except pd.errors.EmptyDataError:
        logger.warning("%s is empty; measuring everything.", path)
        return frozenset()
    except pd.errors.ParserError as exc:
        logger.warning("Could not parse %s (%s); measuring everything.", path, exc)
        return frozenset()
    except KeyError as exc:
        logger.warning("%s is missing expected resume column %s; measuring everything.", path, exc)
        return frozenset()


# First-party band mismatches (this repo's own `_band_mapping` helpers) raise
# the typed `BandIncompatibilityError` and are matched by type in
# `_is_band_incompatibility`. Only a narrow, documented set of third-party
# stems that validate channel count themselves -- and raise a plain
# `ValueError` this repo doesn't control -- fall back to a message match.
# `isinstance(exc, ValueError)` alone is far too wide for that fallback: our
# own `compose_config` and various third-party constructors raise plain
# `ValueError` for reasons that have nothing to do with band compatibility
# (a malformed --config YAML, an unrelated validation failure, ...), so
# matching on type alone would log those as "incompatible with this band
# config" and silently drop them from the CSV instead of failing loudly.
_THIRD_PARTY_CHANNEL_MISMATCH_MARKERS: tuple[str, ...] = (
    # third-party stems that validate channel count themselves (torchgeo's
    # fixed-channel checkpoints, timm patch-embed).  Deliberately *not*
    # included: "images has N channels but src_bands has M entries" from
    # map_to_model_bands, which means the caller passed a tensor disagreeing
    # with its own BandSpecs — a pipeline bug that must stay loud.  It cannot
    # fire here anyway, since n_channels is len(band_specs) by construction.
    "input channels",
    "num_chans",
    "in_chans",
)


def _is_band_incompatibility(exc: BaseException) -> BaseException | None:
    """Return the band-incompatibility cause in *exc*'s chain, or None.

    Model constructors may wrap the band-mismatch error, so the exception
    chain is walked. First-party mismatches are matched by the typed
    ``BandIncompatibilityError``; a narrow, documented set of third-party
    channel-count messages is matched as a fallback. Anything else (a
    missing checkpoint, an exhausted disk quota, a malformed config) is a
    real failure and must propagate rather than be recorded as a skip.
    """
    seen: set[int] = set()
    cause: BaseException | None = exc
    # `__cause__ or __context__` can cycle when an exception is raised while
    # handling itself, so visited frames are tracked rather than trusted to
    # terminate.
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        if isinstance(cause, BandIncompatibilityError):
            return cause
        if isinstance(cause, ValueError):
            message = str(cause).lower()
            if any(marker in message for marker in _THIRD_PARTY_CHANNEL_MISMATCH_MARKERS):
                return cause
        cause = cause.__cause__ or cause.__context__
    return None


def _build_model(
    cfg_model: dict,
    band_specs: list,
    normalization: str,
    band_config: str,
) -> nn.Module | None:
    """Instantiate the model for *band_config*, or None if incompatible.

    Model/band incompatibilities are expected and numerous: several configs
    are RGB-only (a 3-channel pretrained stem) and several are
    multispectral-only (``tgeo_resnet50_s2all_moco`` has a 13-channel stem).
    Those raise the typed ``BandIncompatibilityError`` (first-party) or a
    narrow, documented third-party channel-count ``ValueError`` and are
    skipped with a warning, mirroring ``seg_corruption_pipeline``.
    """
    try:
        return instantiate(cfg_model, bands=band_specs, normalization=normalization)
    except Exception as exc:
        cause = _is_band_incompatibility(exc)
        if cause is None:
            raise
        logger.warning(
            "Skipping %s/%s: model is incompatible with this band config: %s",
            cfg_model.get("name", cfg_model["_target_"]),
            band_config,
            cause,
        )
        return None


def _measure_backbone(
    model: nn.Module,
    n_channels: int,
    image_size: int,
    device: torch.device,
    batch_size: int,
    n_warmup: int,
    n_measure: int,
) -> tuple[dict[str, float | None], int]:
    """Run ``measure_profile`` on a synthetic batch, halving on CUDA OOM.

    Returns the metrics dict and the batch size actually used, so a cell that
    fell back to a smaller batch stays interpretable (GFLOPs is per-sample
    either way; throughput/memory/energy are not).
    """
    while True:
        try:
            x = torch.randn(batch_size, n_channels, image_size, image_size, device=device)
            return measure_profile(
                model, x, device, n_warmup=n_warmup, n_measure=n_measure
            ), batch_size
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 1:
                raise
            batch_size //= 2
            torch.cuda.empty_cache()
            logger.warning("CUDA OOM — retrying at timing_batch_size=%d", batch_size)


def _probe_gflops(
    model: nn.Module,
    n_channels: int,
    image_size: int,
    device: torch.device,
    head: str,
    n_classes: int,
) -> tuple[float | None, float, int]:
    """Build the linear/mlp probe the way ``linear.py`` does and count it.

    ``feature_dim`` is *not* configured anywhere — ``linear.py`` infers it
    from ``X.shape[1]`` of the extracted features.  So the width has to come
    from a real forward pass, and the probe can only be constructed after it.

    Returns ``None`` gflops (with a warning) if ``_count_gflops`` raises its
    typed ``NotImplementedError`` -- i.e. the probe cannot be traced under
    ``torch.utils.flop_counter.FlopCounterMode`` through the public API.
    """
    with torch.inference_mode():
        feats = model(torch.randn(1, n_channels, image_size, image_size, device=device))
    feature_dim = int(feats.shape[1])

    if head == "mlp":
        probe: nn.Module = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim, n_classes, bias=True),
        )
    else:
        probe = nn.Linear(feature_dim, n_classes, bias=True)
    probe.to(device).eval()

    try:
        gflops: float | None = _count_gflops(probe, torch.randn(2, feature_dim, device=device))
    except NotImplementedError as exc:
        logger.warning("[flops] %s", exc)
        gflops = None
    return gflops, _count_params(probe), feature_dim


def _n_tokens(model: nn.Module, image_size: int) -> int | None:
    """Return the number of *patch* tokens, or None for CNN backbones.

    ``forward_patch_features`` returns an already-pooled ``(B, D)`` vector, so
    the token count cannot be read off the model's output.  It is instead
    derived from the patch-embedding grid, which is what actually drives
    attention cost: ``n_tokens ~ (image_size / patch)^2``.

    Prefix tokens (CLS + registers) are deliberately *excluded* — they are
    read via ``num_prefix_tokens`` where a module exposes one, so
    register-token ViTs (DINOv3 reports 5) report the same patch-grid size as
    a plain ViT at equal patch size.
    """
    patch: tuple[int, int] | None = None
    for module in model.modules():
        p = getattr(module, "patch_size", None)
        if isinstance(p, (tuple, list)) and len(p) == 2:
            patch = (int(p[0]), int(p[1]))
            break
        if isinstance(p, int):
            patch = (p, p)
            break

    if patch is None or patch[0] <= 0 or patch[1] <= 0:
        return None
    # Derived from the size actually measured, not the model's configured
    # `grid_size`: the two disagree whenever a backbone is run off its native
    # resolution, and it is the measured grid that drove the FLOPs.
    return (image_size // patch[0]) * (image_size // patch[1])


def _seg_head_gflops(
    probe: nn.Module,
    n_channels: int,
    image_size: int,
    device: torch.device,
) -> float | None:
    """Count the segmentation head alone, mirroring ``SegmentationProbe.forward``.

    Counting the probe end-to-end would be wrong: ``forward`` wraps the frozen
    backbone in ``no_grad`` + ``autocast``, which perturbs counts.  So the
    backbone is run once to populate ``_features``, then only
    ``head(features, H, W)`` is counted.

    Returns ``None`` (with a warning) if the head cannot be traced under
    ``FlopCounterMode`` through the public API alone: the feature maps handed
    to the head come from a backbone forward pass run under
    ``inference_mode``, so they carry no ``grad_fn``, and a head whose own
    module structure needs one (rare, but possible for exotic decoders) trips
    the same ``module_tracker`` assertion ``_count_gflops`` guards against.
    """
    x = torch.randn(1, n_channels, image_size, image_size, device=device)
    probe._features.clear()
    with torch.inference_mode():
        _ = probe.backbone(x)
    features = [probe._process_feature(probe._features[n]) for n in probe.layer_names]

    class _HeadOnly(nn.Module):
        def __init__(self, head: nn.Module, size: tuple[int, int]) -> None:
            super().__init__()
            self.head = head
            self.size = size

        def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
            return self.head(feats, *self.size)

    wrapper = _HeadOnly(probe.head, (image_size, image_size)).to(device).eval()
    # _count_gflops slices sample[:1]; a list of feature maps is already
    # batch-1 here, so hand it straight through.
    from torch.utils.flop_counter import FlopCounterMode

    try:
        with FlopCounterMode(display=False) as counter, torch.inference_mode():
            wrapper(features)
    except AssertionError as exc:
        # 'Expected gradient function to be set' — module_tracker wants a
        # grad_fn on every head input; anything else is a genuine bug.
        if "Expected gradient function" not in str(exc):
            raise
        logger.warning(
            "[flops] %s is incompatible with torch.utils.flop_counter.FlopCounterMode "
            "(module_tracker requires a gradient function on its inputs); "
            "gflops_head will be None.",
            type(probe.head).__name__,
        )
        return None
    return float(counter.get_total_flops()) / 1e9


def _flops_row(
    base_meta: dict,
    model: nn.Module,
    image_size: int,
    *,
    task: str,
    head_type: str = "",
    **values: object,
) -> dict:
    """One compute_cost.csv row; metric slots default to None and are filled per task."""
    row = {
        **base_meta,
        "task": task,
        "head_type": head_type,
        "gflops_backbone": None,
        "gflops_head": None,
        "gflops_probe": None,
        "gflops_total": None,
        "params_backbone_m": None,
        "params_head_m": None,
        "params_probe_m": None,
        "feature_dim": None,
        "pool": getattr(model, "pool", None),
        "n_tokens": _n_tokens(model, image_size),
        "throughput_samples_per_sec": None,
        "latency_ms_per_batch_p50": None,
        "peak_gpu_mem_gb": None,
        "reserved_gpu_mem_gb": None,
        "timing_batch_size": None,
        "measured_at": _now(),
    }
    row.update(values)
    return row


def main(cfg: FlopsSettings) -> None:
    """Measure per-sample compute cost for one model config."""
    output_path = str(cfg.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    device = resolve_configured_device(str(cfg.device))
    image_size = int(cfg.image_size)
    normalization = str(cfg.normalization)
    model_target = str(cfg.model["_target_"])
    model_name = str(cfg.model.get("name", model_target.split(".")[-1]))

    ds_cls = get_bench_dataset_class(str(cfg.band_source))
    bench = ds_cls()
    band_specs_for = {
        "rgb": bench.select_band_specs(bench.rgb_bands),
        "s2": bench.select_band_specs(None),
    }

    completed = _load_completed(output_path) if bool(cfg.resume) else frozenset()

    # Merge the model's eval block over the base eval settings (models only
    # override `layers` / `head_type`).
    seg_eval_cfg = cfg.eval
    model_eval = cfg.model.get("eval")
    if model_eval is not None:
        seg_eval_cfg = merge(seg_eval_cfg, model_eval)
    seg_layers = list(seg_eval_cfg.segmentation.layers)

    rows: list[dict] = []
    n_skipped = 0

    for band_config in list(cfg.band_configs):
        band_specs = band_specs_for[band_config]
        n_channels = len(band_specs)

        model = _build_model(cfg.model, band_specs, normalization, band_config)
        if model is None:
            n_skipped += 1
            continue
        model.to(device).eval()

        base_meta = {
            "model": model_target,
            "name": model_name,
            "band_config": band_config,
            "n_channels": n_channels,
            "image_size": image_size,
            "num_classes": int(cfg.probe_num_classes),
        }

        # --- classification cell ------------------------------------------
        cls_key = (model_name, band_config, "classification", "")
        if cls_key in completed:
            logger.info("Skip (%s, %s, classification) — already done", model_name, band_config)
        else:
            try:
                metrics, used_batch = _measure_backbone(
                    model,
                    n_channels,
                    image_size,
                    device,
                    int(cfg.timing_batch_size),
                    int(cfg.n_warmup),
                    int(cfg.n_measure),
                )
                gflops_backbone = metrics["gflops"]
                gflops_probe, params_probe_m, feature_dim = _probe_gflops(
                    model,
                    n_channels,
                    image_size,
                    device,
                    str(cfg.probe_head),
                    int(cfg.probe_num_classes),
                )
            except Exception as exc:
                # Several wrappers only discover a band mismatch on the first
                # forward pass (their band mapping runs inside
                # `_forward_patch_features`). Same skip one stage later; must
                # not abort — the other band config is usually fine.
                cause = _is_band_incompatibility(exc)
                if cause is None:
                    raise
                logger.warning(
                    "Skipping %s/%s classification: model is incompatible with this "
                    "band config at forward time: %s",
                    model_name,
                    band_config,
                    cause,
                )
                n_skipped += 1
                del model
                _free(device)
                continue
            rows.append(
                _flops_row(
                    base_meta,
                    model,
                    image_size,
                    task="classification",
                    gflops_backbone=gflops_backbone,
                    gflops_probe=gflops_probe,
                    gflops_total=(
                        None
                        if gflops_backbone is None or gflops_probe is None
                        else gflops_backbone + gflops_probe
                    ),
                    params_backbone_m=metrics["params_m"],
                    params_probe_m=params_probe_m,
                    feature_dim=feature_dim,
                    throughput_samples_per_sec=metrics["throughput_samples_per_sec"],
                    latency_ms_per_batch_p50=metrics["latency_ms_per_batch_p50"],
                    peak_gpu_mem_gb=metrics["peak_gpu_mem_gb"],
                    reserved_gpu_mem_gb=metrics["reserved_gpu_mem_gb"],
                    timing_batch_size=used_batch,
                )
            )
            logger.info(
                "%s/%s classification: backbone=%s GF probe=%s GF (D=%d)",
                model_name,
                band_config,
                f"{gflops_backbone:.4f}" if gflops_backbone is not None else "None",
                f"{gflops_probe:.6f}" if gflops_probe is not None else "None",
                feature_dim,
            )

        # --- segmentation cells -------------------------------------------
        seg_band_configs = set(cfg.seg_band_configs)
        if band_config not in seg_band_configs or not seg_layers:
            if band_config in seg_band_configs and not seg_layers:
                logger.info(
                    "No eval.segmentation.layers for %s — skipping segmentation cells", model_name
                )
            del model
            _free(device)
            continue

        for head_type in list(cfg.seg_head_types):
            seg_key = (model_name, band_config, "segmentation", head_type)
            if seg_key in completed:
                logger.info("Skip (%s, %s, %s) — already done", model_name, band_config, head_type)
                continue
            head_cfg = merge(seg_eval_cfg, {"segmentation": {"head_type": head_type}})
            try:
                # build_seg_probe_and_solver runs _dry_run_channels(), a real
                # forward pass, so the model must already be on-device.
                probe, _solver = build_seg_probe_and_solver(
                    model, int(cfg.seg_num_classes), head_cfg.segmentation, device, 1e-3
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning(
                    "Skipping %s/%s seg head=%s: %s", model_name, band_config, head_type, exc
                )
                n_skipped += 1
                continue
            probe.to(device).eval()

            gflops_head = _seg_head_gflops(probe, n_channels, image_size, device)
            rows.append(
                _flops_row(
                    base_meta,
                    model,
                    image_size,
                    task="segmentation",
                    head_type=head_type,
                    num_classes=int(cfg.seg_num_classes),
                    gflops_head=gflops_head,
                    params_backbone_m=_count_params(model),
                    params_head_m=_count_params(probe.head),
                    feature_dim=sum(probe.channels_list),
                )
            )
            logger.info(
                "%s/%s segmentation head=%s: head=%s GF (taps=%s)",
                model_name,
                band_config,
                head_type,
                f"{gflops_head:.4f}" if gflops_head is not None else "None",
                probe.channels_list,
            )
            del probe
            _free(device)

        del model
        _free(device)

    if rows:
        append_rows_atomic(output_path, rows)
    logger.info(
        "Wrote %d rows for %s (%d cells skipped) → %s",
        len(rows),
        model_name,
        n_skipped,
        output_path,
    )


def _now() -> str:
    from datetime import datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
