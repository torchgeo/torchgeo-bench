"""Segmentation Training Task Logic."""

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassCalibrationError,
    MulticlassF1Score,
    MulticlassJaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
)

from .segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
)
from .uq.error_pr import compute_error_pr

logger = logging.getLogger(__name__)

SegMetrics = dict[str, float]
SegImageStatsRow = dict[str, int | float | str | bool]


def _nan() -> float:
    return float("nan")


def _compute_segmentation_image_stats_row(
    *,
    logits: torch.Tensor,
    mask: torch.Tensor,
    image_index: int,
    ignore_index: int,
    num_classes: int,
) -> SegImageStatsRow:
    """Reduce one image worth of logits and mask to a flat statistics row.

    Args:
        logits: Per-image logits of shape ``(C, H, W)``.
        mask: Per-image integer mask of shape ``(H, W)``.
        image_index: Deterministic test-set enumeration index.
        ignore_index: Label value excluded from every metric.
        num_classes: Dataset class count.

    Returns:
        Flat per-image statistics row.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape (C, H, W), got {tuple(logits.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape (H, W), got {tuple(mask.shape)}")

    height, width = int(mask.shape[0]), int(mask.shape[1])
    valid_mask = mask != ignore_index
    valid_pixel_count = int(valid_mask.sum().item())
    ignored_pixel_count = int(mask.numel() - valid_pixel_count)

    row: SegImageStatsRow = {
        "image_index": image_index,
        "height": height,
        "width": width,
        "valid_pixel_count": valid_pixel_count,
        "ignored_pixel_count": ignored_pixel_count,
        "n_gt_classes": 0,
        "n_pred_classes": 0,
        "n_pred_or_gt_classes": 0,
        "image_pixel_accuracy": _nan(),
        "image_miou_gt_present": _nan(),
        "image_miou_pred_or_gt_present": _nan(),
        "mean_1mp": _nan(),
        "median_1mp": _nan(),
        "mean_entropy": _nan(),
        "median_entropy": _nan(),
        "mean_normalized_entropy": _nan(),
        "median_normalized_entropy": _nan(),
        "pixel_error_aupr_1mp": _nan(),
        "pixel_error_auroc_1mp": _nan(),
        "pixel_error_aupr_entropy": _nan(),
        "pixel_error_auroc_entropy": _nan(),
    }
    if valid_pixel_count == 0:
        return row

    valid_logits = logits.permute(1, 2, 0)[valid_mask]
    valid_mask_values = mask[valid_mask].long()

    probs = torch.softmax(valid_logits, dim=1)
    max_prob, pred = probs.max(dim=1)
    one_minus_max_prob = 1.0 - max_prob
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1)
    if num_classes > 1:
        normalized_entropy = entropy / math.log(float(num_classes))
    else:
        normalized_entropy = torch.zeros_like(entropy)

    is_error = pred != valid_mask_values
    pred_counts = torch.bincount(pred, minlength=num_classes)
    gt_counts = torch.bincount(valid_mask_values, minlength=num_classes)
    intersections = torch.bincount(
        valid_mask_values[pred == valid_mask_values],
        minlength=num_classes,
    )
    unions = pred_counts + gt_counts - intersections
    union_present = unions > 0
    gt_present = gt_counts > 0

    iou = torch.full((num_classes,), float("nan"), dtype=torch.float64, device=unions.device)
    iou[union_present] = intersections[union_present].to(torch.float64) / unions[union_present].to(
        torch.float64
    )

    row["n_gt_classes"] = int(gt_present.sum().item())
    row["n_pred_classes"] = int((pred_counts > 0).sum().item())
    row["n_pred_or_gt_classes"] = int(union_present.sum().item())
    row["image_pixel_accuracy"] = float((~is_error).to(torch.float32).mean().item())
    if gt_present.any():
        row["image_miou_gt_present"] = float(iou[gt_present].mean().item())
    if union_present.any():
        row["image_miou_pred_or_gt_present"] = float(iou[union_present].mean().item())

    row["mean_1mp"] = float(one_minus_max_prob.mean().item())
    row["median_1mp"] = float(one_minus_max_prob.median().item())
    row["mean_entropy"] = float(entropy.mean().item())
    row["median_entropy"] = float(entropy.median().item())
    row["mean_normalized_entropy"] = float(normalized_entropy.mean().item())
    row["median_normalized_entropy"] = float(normalized_entropy.median().item())

    error_labels = is_error.to(torch.int64).cpu().numpy()
    if valid_pixel_count >= 2 and np.unique(error_labels).size == 2:
        one_minus_max_prob_np = one_minus_max_prob.to(torch.float64).cpu().numpy()
        entropy_np = entropy.to(torch.float64).cpu().numpy()
        one_minus_max_metrics = compute_error_pr(
            is_error=error_labels,
            uncertainty=one_minus_max_prob_np,
        )
        entropy_metrics = compute_error_pr(
            is_error=error_labels,
            uncertainty=entropy_np,
        )
        row["pixel_error_aupr_1mp"] = float(one_minus_max_metrics["aupr"])
        row["pixel_error_auroc_1mp"] = float(one_minus_max_metrics["auroc"])
        row["pixel_error_aupr_entropy"] = float(entropy_metrics["aupr"])
        row["pixel_error_auroc_entropy"] = float(entropy_metrics["auroc"])

    return row


def _collect_segmentation_image_stats_rows(
    *,
    logits: torch.Tensor,
    masks: torch.Tensor,
    image_index_start: int,
    ignore_index: int,
    num_classes: int,
) -> list[SegImageStatsRow]:
    """Return one statistics row per image in a batch."""
    rows: list[SegImageStatsRow] = []
    for idx in range(int(logits.shape[0])):
        rows.append(
            _compute_segmentation_image_stats_row(
                logits=logits[idx],
                mask=masks[idx],
                image_index=image_index_start + idx,
                ignore_index=ignore_index,
                num_classes=num_classes,
            )
        )
    return rows


class SegmentationSolver:
    """A lightweight trainer for the SegmentationProbe."""

    def __init__(
        self,
        model: SegmentationProbe,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cuda",
        criterion: nn.Module | None = None,
        lr_scheduler: str = "cosine",
        ignore_index: int = 255,
        n_bins_ece: int = 15,
    ) -> None:
        """Initialize the SegmentationSolver.

        Args:
            model: The SegmentationProbe model to train.
            num_classes: Number of segmentation classes.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            device: Device to run training on ('cuda' or 'cpu').
            criterion: Loss module. Defaults to CrossEntropyLoss with ignore_index.
            lr_scheduler: LR schedule: "cosine" (CosineAnnealingLR) or "none" (constant LR).
            ignore_index: Label value to ignore in loss and metrics (default: 255).
            n_bins_ece: Number of bins for pixel-level ECE (default: 15).
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler

        self.ignore_index = ignore_index
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.criterion = (
            criterion
            if criterion is not None
            else nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        )

        self.metric = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_fw_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="weighted",
        )
        self.metric_precision = MulticlassPrecision(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_recall = MulticlassRecall(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_f1 = MulticlassF1Score(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average="macro",
        )
        self.metric_ece = MulticlassCalibrationError(
            num_classes=self.num_classes,
            n_bins=n_bins_ece,
            ignore_index=self.ignore_index,
        )
        self._all_metrics = [
            self.metric,
            self.metric_fw_iou,
            self.metric_precision,
            self.metric_recall,
            self.metric_f1,
            self.metric_ece,
        ]

        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.device_type = torch.device(device).type

    def _make_scheduler(self, epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Return a CosineAnnealingLR scheduler, or None for constant LR."""
        if self.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        if self.lr_scheduler_type == "none":
            return None
        raise ValueError(
            f"Unknown lr_scheduler {self.lr_scheduler_type!r}. Expected 'cosine' or 'none'."
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> float | None:
        """Train the segmentation probe.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader for per-epoch mIoU logging.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.

        Returns:
            Val mIoU from the final epoch if val_loader is given, else None.
        """
        scheduler = self._make_scheduler(epochs)
        last_val_miou: float | None = None

        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            total_loss = 0.0

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = track(train_loader, description=desc) if verbose else train_loader
            for _num_batches, batch in enumerate(batches, start=1):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device).long()
                else:
                    images, masks = batch[0].to(self.device), batch[1].to(self.device).long()

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                last_val_miou = val_metrics["mIoU"]
                if verbose:
                    logger.info(f"Epoch {epoch + 1} Val mIoU: {last_val_miou:.4f}")

        return last_val_miou

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        collect_preds: bool = False,
        collect_image_stats: bool = False,
    ) -> (
        "SegMetrics"
        " | tuple[SegMetrics, torch.Tensor]"
        " | tuple[SegMetrics, list[SegImageStatsRow]]"
        " | tuple[SegMetrics, torch.Tensor, list[SegImageStatsRow]]"
    ):
        """Evaluate the model on a dataloader and return segmentation metrics.

        Args:
            dataloader: Evaluation data loader.
            collect_preds: If True, also return predicted class maps (N, H, W) int64.
            collect_image_stats: If True, also return one row per test image.

        Returns:
            Dict of metric name → value, or a tuple that additionally contains
            predictions and/or per-image statistics depending on the collection
            flags.
        """
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        pred_list: list[torch.Tensor] = []
        image_stats_rows: list[SegImageStatsRow] = []
        image_index = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
            else:
                images, masks = batch[0].to(self.device), batch[1].to(self.device)

            # Ensure masks are (B, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = masks.long()

            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model(images)

            for m in self._all_metrics:
                m.update(logits, masks)

            if collect_preds:
                pred_list.append(logits.argmax(dim=1).cpu())
            if collect_image_stats:
                image_stats_rows.extend(
                    _collect_segmentation_image_stats_rows(
                        logits=logits,
                        masks=masks,
                        image_index_start=image_index,
                        ignore_index=self.ignore_index,
                        num_classes=self.num_classes,
                    )
                )
            image_index += int(masks.shape[0])

        metrics = self._compute_metrics()
        if collect_preds and collect_image_stats:
            return metrics, torch.cat(pred_list, dim=0), image_stats_rows
        if collect_preds:
            return metrics, torch.cat(pred_list, dim=0)
        if collect_image_stats:
            return metrics, image_stats_rows
        return metrics

    def fit_cached(
        self,
        train_cache: CachedFeaturesDataset,
        val_cache: CachedFeaturesDataset | None = None,
        batch_size: int = 64,
        epochs: int = 10,
        verbose: bool = True,
        gpu_train: "GPUTensorCache | None" = None,
        gpu_val: "GPUTensorCache | None" = None,
    ) -> float | None:
        """Train the segmentation head on pre-cached backbone features.

        The backbone is **not** called during training — cached features are fed
        directly to ``self.model.head``, which is the only component that runs
        a forward/backward pass.

        The entire feature cache is pre-moved to the GPU as contiguous tensors
        (:class:`GPUTensorCache`), eliminating per-batch CPU→GPU DMA transfers
        and ``torch.stack`` calls.

        If ``gpu_train`` is provided, that pre-built cache is used directly,
        allowing callers (e.g. an HPO loop) to transfer the cache once and
        reuse it across many calls.

        Args:
            train_cache: Pre-extracted training features from
                :meth:`SegmentationProbe.extract_segmentation_features`.
            val_cache: Optional validation cache for per-epoch mIoU logging.
            batch_size: Batch size for iterating over cached data.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.
            gpu_train: Optional pre-built GPU cache for training. If provided,
                the GPU transfer is skipped.
            gpu_val: Optional pre-built GPU cache for validation. Used only
                when ``gpu_train`` is also provided.

        Returns:
            Val mIoU from the final epoch if val_cache is given, else None.
        """
        if gpu_train is None:
            gpu_train = GPUTensorCache.from_cached(train_cache, self.device)
            if val_cache is not None:
                gpu_val = GPUTensorCache.from_cached(val_cache, self.device)

        # Fast path: GPU tensor cache — no DataLoader, no host→device transfer per batch
        scheduler = self._make_scheduler(epochs)

        input_hw: tuple[int, int] = (gpu_train.masks.shape[-2], gpu_train.masks.shape[-1])
        last_val_miou: float | None = None
        num_batches = math.ceil(len(gpu_train) / batch_size)

        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            total_loss = 0.0
            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = gpu_train.shuffled_batches(batch_size)
            batches = track(batches, total=num_batches, description=desc) if verbose else batches
            for features, masks in batches:
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model.head(features, *input_hw)
                    loss = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            if gpu_val is not None:
                val_metrics = self._evaluate_gpu_cache(gpu_val, batch_size)
                last_val_miou = val_metrics["mIoU"]
                if verbose:
                    logger.info(f"Epoch {epoch + 1} Val mIoU: {last_val_miou:.4f}")

        return last_val_miou

    def evaluate_cached(
        self,
        cache: CachedFeaturesDataset,
        batch_size: int = 64,
        collect_preds: bool = False,
        collect_image_stats: bool = False,
    ) -> (
        "SegMetrics"
        " | tuple[SegMetrics, torch.Tensor]"
        " | tuple[SegMetrics, list[SegImageStatsRow]]"
        " | tuple[SegMetrics, torch.Tensor, list[SegImageStatsRow]]"
    ):
        """Evaluate on a CachedFeaturesDataset.

        The cache is moved to GPU as a :class:`GPUTensorCache` for zero
        per-batch host→device transfers.

        Args:
            cache: Pre-extracted features (output of
                :meth:`SegmentationProbe.extract_segmentation_features`).
            batch_size: Batch size for iterating over the cache.
            collect_preds: If True, also return predicted class maps (N, H, W) int64.
            collect_image_stats: If True, also return one row per cached image.

        Returns:
            Dict of metric name → value, or a tuple that additionally contains
            predictions and/or per-image statistics depending on the collection
            flags.
        """
        gpu_cache = GPUTensorCache.from_cached(cache, self.device)
        return self._evaluate_gpu_cache(
            gpu_cache,
            batch_size,
            collect_preds=collect_preds,
            collect_image_stats=collect_image_stats,
        )

    def _compute_metrics(self) -> "SegMetrics":
        """Compute and return all metrics as a dict."""
        return {
            "mIoU": self.metric.compute().item(),
            "fw_IoU": self.metric_fw_iou.compute().item(),
            "precision": self.metric_precision.compute().item(),
            "recall": self.metric_recall.compute().item(),
            "f1": self.metric_f1.compute().item(),
            "pixel_ece": self.metric_ece.compute().item(),
        }

    @torch.no_grad()
    def _evaluate_gpu_cache(
        self,
        gpu_cache: GPUTensorCache,
        batch_size: int,
        collect_preds: bool = False,
        collect_image_stats: bool = False,
    ) -> (
        "SegMetrics"
        " | tuple[SegMetrics, torch.Tensor]"
        " | tuple[SegMetrics, list[SegImageStatsRow]]"
        " | tuple[SegMetrics, torch.Tensor, list[SegImageStatsRow]]"
    ):
        """Evaluate on a :class:`GPUTensorCache` and return segmentation metrics."""
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        pred_list: list[torch.Tensor] = []
        image_stats_rows: list[SegImageStatsRow] = []
        image_index = 0

        input_hw = (gpu_cache.masks.shape[-2], gpu_cache.masks.shape[-1])
        for features, masks in gpu_cache.ordered_batches(batch_size):
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model.head(features, *input_hw)
            for m in self._all_metrics:
                m.update(logits, masks)
            if collect_preds:
                pred_list.append(logits.argmax(dim=1).cpu())
            if collect_image_stats:
                image_stats_rows.extend(
                    _collect_segmentation_image_stats_rows(
                        logits=logits,
                        masks=masks,
                        image_index_start=image_index,
                        ignore_index=self.ignore_index,
                        num_classes=self.num_classes,
                    )
                )
            image_index += int(masks.shape[0])

        metrics = self._compute_metrics()
        if collect_preds and collect_image_stats:
            return metrics, torch.cat(pred_list, dim=0), image_stats_rows
        if collect_preds:
            return metrics, torch.cat(pred_list, dim=0)
        if collect_image_stats:
            return metrics, image_stats_rows
        return metrics
