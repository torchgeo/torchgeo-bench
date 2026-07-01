"""Qualitative UQ visualization pipeline for the segmentation subsample sweep.

Generates a grid PNG per (model, dataset) showing how predictions and
per-pixel uncertainty (normalized entropy) evolve across training fractions.

Grid layout:
  rows = N test images (highest-entropy at the largest fraction)
  left margin = [RGB image | GT mask] shown once per row
  columns = one per training fraction; each cell stacks [Pred mask / Entropy heatmap]
"""

import copy
import logging
import math
import os
import warnings

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.main import _resolve_segmentation_ignore_index
from torchgeo_bench.sample_size_pipeline import _compute_epochs, _subsample_cache
from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
)
from torchgeo_bench.segmentation_task import SegmentationSolver
from torchgeo_bench.segmentation_viz import render_uq_subsample_grid

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)

_TARGET_GRAD_STEPS = 1000  # matches sample_size_config default


@torch.no_grad()
def _infer_subset(
    solver: SegmentationSolver,
    gpu_cache: GPUTensorCache,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference on a subset of a GPUTensorCache.

    Returns:
        preds: (N_sub, H, W) int64 predicted class maps on CPU.
        norm_entropy: (N_sub, H, W) float32 normalized entropy maps on CPU.
    """
    solver.model.eval()
    idx_t = torch.from_numpy(indices.astype(np.int64))
    sub_layers = [t[idx_t] for t in gpu_cache.layer_tensors]
    sub_masks = gpu_cache.masks[idx_t]

    if not gpu_cache._on_device:
        sub_layers = [t.to(gpu_cache.device, non_blocking=True) for t in sub_layers]
        sub_masks = sub_masks.to(gpu_cache.device, non_blocking=True)

    input_hw = (sub_masks.shape[-2], sub_masks.shape[-1])
    pred_list: list[torch.Tensor] = []
    ent_list: list[torch.Tensor] = []

    for start in range(0, len(idx_t), batch_size):
        s = slice(start, start + batch_size)
        feats = [t[s] for t in sub_layers]
        with torch.autocast(device_type=solver.device_type, enabled=solver.use_amp):
            logits = solver.model.head(feats, *input_hw)  # (B, C, H, W)

        probs = torch.softmax(logits.float(), dim=1)
        preds = logits.argmax(dim=1).cpu()

        log_n = math.log(max(solver.num_classes, 2))
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=1) / log_n  # normalized, (B, H, W)

        pred_list.append(preds)
        ent_list.append(entropy.cpu())

    return torch.cat(pred_list, dim=0), torch.cat(ent_list, dim=0)


def _fit_probe(
    probe: SegmentationProbe,
    train_sub_cache: CachedFeaturesDataset,
    val_cache: CachedFeaturesDataset,
    solver_kwargs: dict,
    batch_size: int,
    epochs: int,
) -> SegmentationSolver:
    """Fresh-init probe head, train, return fitted solver."""
    fresh_probe = copy.deepcopy(probe)
    for m in fresh_probe.head.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    solver = SegmentationSolver(**solver_kwargs, model=fresh_probe)
    solver.fit_cached(train_sub_cache, val_cache=val_cache, batch_size=batch_size, epochs=epochs, verbose=False)
    return solver


@hydra.main(config_path="conf", config_name="seg_viz_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate qualitative UQ grid PNGs for the segmentation subsample sweep."""
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    device = torch.device(str(cfg.device))
    model_target = str(cfg.model._target_)
    model_name = str(cfg.model.get("name", model_target.split(".")[-1]))
    dataset_names = list(cfg.dataset.names)
    fractions = [float(f) for f in cfg.seg_viz.fractions]
    viz_seed = int(cfg.seg_viz.seed)
    n_samples = int(cfg.seg_viz.n_samples)
    out_dir = str(cfg.seg_viz.out_dir)
    bands = str(cfg.dataset.bands)
    partition = str(cfg.dataset.partition)
    normalization = str(cfg.dataset.get("normalization", "bandspec_zscore"))

    for dataset_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(dataset_name)
        except KeyError:
            logger.warning("Skipping unknown dataset %s", dataset_name)
            continue

        if ds_cls.task != "segmentation":
            logger.warning("Skipping %s — not a segmentation dataset (task=%s)", dataset_name, ds_cls.task)
            continue

        loaded = get_datasets(
            dataset_name=dataset_name,
            partition_name=partition,
            batch_size=int(cfg.dataset.batch_size),
            num_workers=int(cfg.dataset.get("num_workers", 4)),
            return_val=True,
            image_size=getattr(cfg.dataset, "image_size", None),
            interpolation=str(cfg.dataset.get("interpolation", "bilinear")),
            bands=bands,
        )
        _, train_loader, val_loader, test_loader = loaded

        bench = ds_cls()
        bands_resolved = (
            tuple(bench.rgb_bands)
            if bands == "rgb"
            else None
            if bands in ("all", None)
            else tuple(bands)
        )
        band_specs = bench.select_band_specs(bands_resolved)

        try:
            model = instantiate(
                cfg.model, bands=band_specs, normalization=normalization, _convert_="object"
            )
        except ValueError as exc:
            logger.warning("Skipping %s/%s: model instantiation failed: %s", model_name, dataset_name, exc)
            continue
        model.to(device).eval()

        # Merge model-specific eval config (e.g. resnet50 ships its own FPN layers)
        seg_eval_cfg = cfg.eval
        if "eval" in cfg.model and cfg.model.eval is not None:
            seg_eval_cfg = OmegaConf.merge(seg_eval_cfg, cfg.model.eval)
        seg_cfg = seg_eval_cfg.segmentation

        if not list(seg_cfg.layers):
            logger.warning(
                "Skipping %s/%s: eval.segmentation.layers is not set.",
                model_name,
                dataset_name,
            )
            continue

        criterion_template = instantiate(seg_cfg.criterion)
        ignore_index = _resolve_segmentation_ignore_index(seg_cfg, criterion_template)
        batch_size = int(seg_cfg.batch_size)
        num_classes = bench.num_classes

        # Resolve RGB channel indices relative to the *loaded* band subset.
        # bench.rgb_indices indexes into the full band list, which is wrong when
        # only a subset (e.g. rgb-only) is loaded. Recompute against band_specs.
        loaded_names = [s.name for s in band_specs]
        rgb_indices = [loaded_names.index(n) for n in bench.rgb_bands if n in loaded_names]
        if len(rgb_indices) != 3:
            rgb_indices = [0, 1, 2]

        logger.info("Extracting backbone features for %s / %s ...", model_name, dataset_name)
        seg_probe = SegmentationProbe(
            backbone=model,
            layer_names=list(seg_cfg.layers),
            num_classes=num_classes,
            head_type=str(seg_cfg.head_type),
            freeze_backbone=True,
        )
        train_cache = seg_probe.extract_segmentation_features(train_loader)
        val_cache = seg_probe.extract_segmentation_features(val_loader)
        test_cache = seg_probe.extract_segmentation_features(test_loader)

        n_train_full = len(train_cache)
        n_test = len(test_cache)

        # Collect raw test images for rendering (need original images, not features)
        logger.info("Collecting test images for rendering ...")
        all_images: list[torch.Tensor] = []
        all_gt: list[torch.Tensor] = []
        for batch in test_loader:
            if isinstance(batch, dict):
                imgs = batch["image"]
                masks = batch["mask"]
            else:
                imgs, masks = batch[0], batch[1]
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            all_images.append(imgs.cpu())
            all_gt.append(masks.cpu().long())
        test_images = torch.cat(all_images, dim=0)   # (N, C, H, W)
        test_gt = torch.cat(all_gt, dim=0)           # (N, H, W)

        solver_kwargs = dict(
            num_classes=num_classes,
            lr=float(seg_cfg.lr),
            device=str(device),
            criterion=copy.deepcopy(criterion_template),
            lr_scheduler=str(seg_cfg.get("lr_scheduler", "cosine")),
            ignore_index=ignore_index,
        )

        # --- Step 1: determine anchor images using the largest fraction ---
        anchor_fraction = max(fractions)
        n_anchor = max(1, int(math.floor(n_train_full * anchor_fraction)))
        rng = np.random.default_rng(viz_seed)
        anchor_train_idx = rng.choice(n_train_full, size=n_anchor, replace=False)
        anchor_sub = _subsample_cache(train_cache, anchor_train_idx)
        anchor_epochs = _compute_epochs(n_anchor, batch_size, _TARGET_GRAD_STEPS)

        logger.info("Fitting anchor probe at fraction=%.2f to select high-entropy images ...", anchor_fraction)
        anchor_solver = _fit_probe(
            seg_probe, anchor_sub, val_cache, solver_kwargs, batch_size, anchor_epochs
        )
        anchor_gpu = GPUTensorCache.from_cached(test_cache, str(device))
        test_idx_all = np.arange(n_test)
        _, anchor_entropy = _infer_subset(anchor_solver, anchor_gpu, test_idx_all, batch_size)
        # Select images with highest mean entropy
        mean_ent = anchor_entropy.mean(dim=(1, 2)).numpy()
        anchor_indices = np.argsort(mean_ent)[::-1][:n_samples].copy()
        logger.info("Selected %d high-entropy test images.", len(anchor_indices))

        # --- Step 2: fit one probe per fraction, collect preds + entropy for anchor images ---
        fraction_results: list[dict] = []
        for frac in fractions:
            n_sub = max(1, int(math.floor(n_train_full * frac)))
            rng_f = np.random.default_rng(viz_seed)
            train_idx = rng_f.choice(n_train_full, size=n_sub, replace=False)
            sub_cache = _subsample_cache(train_cache, train_idx)
            epochs = _compute_epochs(n_sub, batch_size, _TARGET_GRAD_STEPS)

            logger.info("Fitting probe at fraction=%.2f (%d samples, %d epochs) ...", frac, n_sub, epochs)
            solver_f = _fit_probe(seg_probe, sub_cache, val_cache, solver_kwargs, batch_size, epochs)
            gpu_cache_f = GPUTensorCache.from_cached(test_cache, str(device))
            preds_f, entropy_f = _infer_subset(solver_f, gpu_cache_f, anchor_indices, batch_size)
            fraction_results.append({"fraction": frac, "preds": preds_f, "entropy": entropy_f})

        # --- Step 3: render and save ---
        anchor_images = test_images[anchor_indices]
        anchor_gt = test_gt[anchor_indices]

        grid = render_uq_subsample_grid(
            images=anchor_images,
            gt_masks=anchor_gt,
            fraction_results=fraction_results,
            num_classes=num_classes,
            rgb_indices=rgb_indices,
            ignore_index=ignore_index,
        )

        try:
            from PIL import Image as PILImage
        except ImportError as e:
            raise ImportError(
                "Pillow is required for segmentation visualization. "
                "Install it with: pip install torchgeo-bench[viz]"
            ) from e

        dest = os.path.join(out_dir, model_name)
        os.makedirs(dest, exist_ok=True)
        out_path = os.path.join(dest, f"{dataset_name}_uq_grid.png")
        PILImage.fromarray(grid).save(out_path)
        logger.info("Saved UQ subsample grid → %s", out_path)


if __name__ == "__main__":
    main()
