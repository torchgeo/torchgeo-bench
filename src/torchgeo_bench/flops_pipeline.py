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

import pandas as pd
import torch
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from torch import nn

from torchgeo_bench.bands import BandCompatibilityError
from torchgeo_bench.config import instantiate
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.model_profile import (
    _count_gflops,
    _count_params,
    measure_profile,
)
from torchgeo_bench.results import append_rows_atomic
from torchgeo_bench.segmentation_task import build_seg_probe_and_solver
from torchgeo_bench.utils import resolve_device

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)

# TerraMind carries the band configuration in the *model config*, not just in
# the tensor shape: TerraTorchTerraMindBench takes both `bands` and `modality`,
# and the modality selects which pretrained tokenizer (and which band table and
# normalization statistics) the input is mapped through.  The `_rgb` and
# S2L2A configs are therefore one model at two points on the band axis, not two
# models — they are emitted under the merged name with `band_config` telling
# them apart.  Each config is only ever measured at its matching band config.
_MODALITY_FOR_BAND_CONFIG = {"rgb": "RGB", "s2": "S2L2A"}
_TERRAMIND_MERGED_NAME = {
    "tt_terramind_v1_base": "tt_terramind_v1_base",
    "tt_terramind_v1_base_rgb": "tt_terramind_v1_base",
    "tt_terramind_v1_large": "tt_terramind_v1_large",
    "tt_terramind_v1_large_rgb": "tt_terramind_v1_large",
}


def _load_completed(path: str) -> frozenset[tuple]:
    """Return the ``(name, band_config, task, head_type)`` keys already in *path*."""
    if not os.path.exists(path):
        return frozenset()
    with FileLock(f"{path}.lock"):
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


def _is_terramind(cfg_model: DictConfig) -> bool:
    return "TerraMind" in str(cfg_model._target_)


def _build_model(
    cfg_model: DictConfig,
    band_specs: list,
    normalization: str,
    band_config: str,
) -> nn.Module | None:
    """Instantiate the model, skipping only explicitly incompatible band selections."""
    if _is_terramind(cfg_model):
        # Each TerraMind config declares its own modality and is only ever
        # measured at the matching band config (enforced by the caller), so
        # the declared modality is used as-is rather than overridden.  Assert
        # the agreement here too: pairing S2RGB with 12 channels is the one
        # failure mode that yields a plausible-looking wrong number instead of
        # an exception.
        declared = str(cfg_model.get("modality", "S2L2A"))
        expected = _MODALITY_FOR_BAND_CONFIG[band_config]
        if declared != expected:
            raise ValueError(
                f"TerraMind modality {declared!r} does not match band config "
                f"{band_config!r} ({len(band_specs)} channels, expects {expected!r}). "
                f"Measuring this pair would map through the wrong band table."
            )
    try:
        return instantiate(cfg_model, bands=band_specs, normalization=normalization)
    except BandCompatibilityError as exc:
        logger.warning(
            "Skipping %s/%s: model is incompatible with this band config: %s",
            cfg_model.get("name", cfg_model._target_),
            band_config,
            exc,
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
) -> tuple[float, float, int]:
    """Build the linear/mlp probe the way ``linear.py`` does and count it.

    ``feature_dim`` is *not* configured anywhere — ``linear.py`` infers it
    from ``X.shape[1]`` of the extracted features.  So the width has to come
    from a real forward pass, and the probe can only be constructed after it.
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
    elif head == "linear":
        probe = nn.Linear(feature_dim, n_classes, bias=True)
    else:
        raise ValueError(f"Unknown probe head {head!r}; expected 'linear' or 'mlp'.")
    probe.to(device).eval()

    gflops = _count_gflops(probe, torch.randn(2, feature_dim, device=device))
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
) -> float:
    """Count the segmentation head alone, mirroring ``SegmentationProbe.forward``.

    Counting the probe end-to-end would be wrong: ``forward`` wraps the frozen
    backbone in ``no_grad`` + ``autocast``, which perturbs counts.  So the
    backbone is run once to populate ``_features``, then only
    ``head(features, H, W)`` is counted.
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
    return _count_gflops(wrapper, features)


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
        "lenient_grad_hooks": False,
        "measured_at": _now(),
    }
    row.update(values)
    return row


def main(cfg: DictConfig) -> None:
    """Measure per-sample compute cost for one model config."""
    output_path = str(cfg.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    device = resolve_device(str(cfg.device))
    image_size = int(cfg.image_size)
    normalization = str(cfg.normalization)
    model_target = str(cfg.model._target_)
    config_name = str(cfg.model.get("name", model_target.split(".")[-1]))
    # TerraMind's _rgb / non-_rgb configs are the same model at two points on
    # the band axis, so they report under one merged name.
    model_name = _TERRAMIND_MERGED_NAME.get(config_name, config_name)

    ds_cls = get_bench_dataset_class(str(cfg.band_source))
    bench = ds_cls()
    band_specs_for = {
        "rgb": bench.select_band_specs(bench.rgb_bands),
        "s2": bench.select_band_specs(None),
    }

    completed = _load_completed(output_path) if bool(cfg.resume) else frozenset()

    # Merge the model's eval block over the base eval config (models only
    # override `layers` / `head_type`). Struct mode comes off first: model
    # configs carry eval keys this pipeline ignores (e.g. `c_range`), which
    # would otherwise raise ConfigKeyError on merge.
    seg_eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg.eval, resolve=True))
    OmegaConf.set_struct(seg_eval_cfg, False)
    if "eval" in cfg.model and cfg.model.eval is not None:
        seg_eval_cfg = OmegaConf.merge(seg_eval_cfg, cfg.model.eval)
    seg_layers = list(seg_eval_cfg.segmentation.get("layers", []))

    n_written = 0
    n_skipped = 0

    for band_config in list(cfg.band_configs):
        # A TerraMind config only ever measures the band config its declared
        # modality matches; the sibling `_rgb` / S2L2A config covers the other.
        if _is_terramind(cfg.model):
            declared = str(cfg.model.get("modality", "S2L2A"))
            if declared != _MODALITY_FOR_BAND_CONFIG[band_config]:
                continue

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
            except BandCompatibilityError as exc:
                logger.warning(
                    "Skipping %s/%s classification: model is incompatible with this "
                    "band config at forward time: %s",
                    model_name,
                    band_config,
                    exc,
                )
                n_skipped += 1
                del model
                _free(device)
                continue
            row = _flops_row(
                base_meta,
                model,
                image_size,
                task="classification",
                gflops_backbone=gflops_backbone,
                gflops_probe=gflops_probe,
                gflops_total=(None if gflops_backbone is None else gflops_backbone + gflops_probe),
                params_backbone_m=metrics["params_m"],
                params_probe_m=params_probe_m,
                feature_dim=feature_dim,
                throughput_samples_per_sec=metrics["throughput_samples_per_sec"],
                latency_ms_per_batch_p50=metrics["latency_ms_per_batch_p50"],
                peak_gpu_mem_gb=metrics["peak_gpu_mem_gb"],
                reserved_gpu_mem_gb=metrics["reserved_gpu_mem_gb"],
                timing_batch_size=used_batch,
            )
            append_rows_atomic(output_path, [row])
            n_written += 1
            logger.info(
                "%s/%s classification: backbone=%s GF probe=%.6f GF (D=%d)",
                model_name,
                band_config,
                f"{gflops_backbone:.4f}" if gflops_backbone is not None else "None",
                gflops_probe,
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
            head_cfg = OmegaConf.merge(
                seg_eval_cfg,
                OmegaConf.create({"segmentation": {"head_type": head_type}}),
            )
            probe, _solver = build_seg_probe_and_solver(
                model, int(cfg.seg_num_classes), head_cfg, device, 1e-3
            )
            probe.to(device).eval()

            gflops_head = _seg_head_gflops(probe, n_channels, image_size, device)
            row = _flops_row(
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
            append_rows_atomic(output_path, [row])
            n_written += 1
            logger.info(
                "%s/%s segmentation head=%s: head=%.4f GF (taps=%s)",
                model_name,
                band_config,
                head_type,
                gflops_head,
                probe.channels_list,
            )
            del probe
            _free(device)

        del model
        _free(device)

    logger.info(
        "Wrote %d rows for %s (%d cells skipped) → %s",
        n_written,
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
