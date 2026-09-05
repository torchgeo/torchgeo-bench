"""Segmentation Training Task Logic."""

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from rich.progress import track
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from omegaconf import DictConfig
from torchmetrics.classification import (
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

logger = logging.getLogger(__name__)

SegMetrics = dict[str, float]


class SegmentationSolver:
    """A lightweight trainer for the SegmentationProbe."""

    def __init__(  # noqa: PLR0913 -- Public constructor options.
        self,
        model: SegmentationProbe,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cuda",
        criterion: nn.Module | None = None,
        lr_scheduler: str = "cosine",
        ignore_index: int = 255,
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
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler
        self.val_history: list[float] = []

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
        self._all_metrics = [
            self.metric,
            self.metric_fw_iou,
            self.metric_precision,
            self.metric_recall,
            self.metric_f1,
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
        *,
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
        self.val_history = []

        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = track(train_loader, description=desc) if verbose else train_loader
            for batch in batches:
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
                    loss: torch.Tensor = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if scheduler is not None:
                scheduler.step()

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                assert isinstance(val_metrics, dict)
                last_val_miou = val_metrics["mIoU"]
                self.val_history.append(last_val_miou)
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch + 1, last_val_miou)

        return last_val_miou

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        collect_preds: bool = False,
        collect_confusions: bool = False,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor] | tuple[SegMetrics, torch.Tensor, torch.Tensor]":
        """Evaluate the model on a dataloader and return segmentation metrics.

        Args:
            dataloader: Evaluation data loader.
            collect_preds: If True, also return predicted class maps (N, H, W) int64.
            collect_confusions: If True, also return per-image confusion matrices.

        Returns:
            Metrics, optionally followed by predictions and/or per-image confusion matrices.
        """
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        pred_list: list[torch.Tensor] = []
        confusion_list: list[torch.Tensor] = []

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

            if collect_preds or collect_confusions:
                predictions = logits.argmax(dim=1)
                if collect_preds:
                    pred_list.append(predictions.cpu())
                if collect_confusions:
                    confusion_list.append(self._per_image_confusions(predictions, masks).cpu())

        metrics = self._compute_metrics()
        return self._evaluation_result(
            metrics,
            pred_list,
            confusion_list,
            collect_preds=collect_preds,
            collect_confusions=collect_confusions,
        )

    def fit_cached(
        self,
        train_cache: "CachedFeaturesDataset | GPUTensorCache",
        val_cache: "CachedFeaturesDataset | GPUTensorCache | None" = None,
        batch_size: int = 64,
        epochs: int = 10,
        *,
        verbose: bool = True,
    ) -> float | None:
        """Train the segmentation head on pre-cached backbone features.

        The backbone is **not** called during training — cached features are fed
        directly to ``self.model.head``, which is the only component that runs
        a forward/backward pass.

        The entire feature cache is pre-moved to the GPU as contiguous tensors
        (:class:`GPUTensorCache`), eliminating per-batch CPU→GPU DMA transfers
        and ``torch.stack`` calls.

        Args:
            train_cache: Pre-extracted training features from
                :meth:`SegmentationProbe.extract_segmentation_features`, or a GPU cache.
            val_cache: Optional validation cache for per-epoch mIoU logging.
            batch_size: Batch size for iterating over cached data.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.

        Returns:
            Val mIoU from the final epoch if val_cache is given, else None.
        """
        gpu_train = (
            train_cache
            if isinstance(train_cache, GPUTensorCache)
            else GPUTensorCache.from_cached(train_cache, self.device)
        )
        gpu_val = (
            GPUTensorCache.from_cached(val_cache, self.device)
            if isinstance(val_cache, CachedFeaturesDataset)
            else val_cache
        )

        # Fast path: GPU tensor cache — no DataLoader, no host→device transfer per batch
        scheduler = self._make_scheduler(epochs)

        input_hw: tuple[int, int] = (gpu_train.masks.shape[-2], gpu_train.masks.shape[-1])
        last_val_miou: float | None = None
        self.val_history = []
        num_batches = math.ceil(len(gpu_train) / batch_size)

        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = gpu_train.shuffled_batches(batch_size)
            batches = track(batches, total=num_batches, description=desc) if verbose else batches
            for features, masks in batches:
                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                    logits = self.model.head(features, *input_hw)
                    loss: torch.Tensor = self.criterion(logits, masks)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            if scheduler is not None:
                scheduler.step()

            if gpu_val is not None:
                val_metrics = self._evaluate_gpu_cache(gpu_val, batch_size)
                assert isinstance(val_metrics, dict)
                last_val_miou = val_metrics["mIoU"]
                self.val_history.append(last_val_miou)
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch + 1, last_val_miou)

        return last_val_miou

    def evaluate_cached(
        self,
        cache: CachedFeaturesDataset,
        batch_size: int = 64,
        *,
        collect_preds: bool = False,
        collect_confusions: bool = False,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor] | tuple[SegMetrics, torch.Tensor, torch.Tensor]":
        """Evaluate on a CachedFeaturesDataset.

        The cache is moved to GPU as a :class:`GPUTensorCache` for zero
        per-batch host→device transfers.

        Args:
            cache: Pre-extracted features (output of
                :meth:`SegmentationProbe.extract_segmentation_features`).
            batch_size: Batch size for iterating over the cache.
            collect_preds: If True, also return predicted class maps (N, H, W) int64.
            collect_confusions: If True, also return per-image confusion matrices.

        Returns:
            Metrics, optionally followed by predictions and/or per-image confusion matrices.
        """
        gpu_cache = GPUTensorCache.from_cached(cache, self.device)
        return self._evaluate_gpu_cache(
            gpu_cache,
            batch_size,
            collect_preds=collect_preds,
            collect_confusions=collect_confusions,
        )

    def _per_image_confusions(self, predictions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Return one confusion matrix per image, excluding ignored labels."""
        predictions = predictions.reshape(len(predictions), -1)
        masks = masks.reshape(len(masks), -1)
        valid = masks != self.ignore_index
        offsets = torch.arange(len(masks), device=masks.device).unsqueeze(1)
        indices = offsets * self.num_classes**2 + masks * self.num_classes + predictions
        indices = indices[valid]
        counts = torch.bincount(indices, minlength=len(masks) * self.num_classes**2)
        return counts.reshape(len(masks), self.num_classes, self.num_classes)

    @staticmethod
    def _evaluation_result(
        metrics: SegMetrics,
        pred_list: list[torch.Tensor],
        confusion_list: list[torch.Tensor],
        *,
        collect_preds: bool,
        collect_confusions: bool,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor] | tuple[SegMetrics, torch.Tensor, torch.Tensor]":
        if collect_preds and collect_confusions:
            return metrics, torch.cat(pred_list, dim=0), torch.cat(confusion_list, dim=0)
        if collect_preds:
            return metrics, torch.cat(pred_list, dim=0)
        if collect_confusions:
            return metrics, torch.cat(confusion_list, dim=0)
        return metrics

    def _compute_metrics(self) -> "SegMetrics":
        """Compute and return all metrics as a dict."""
        return {
            "mIoU": self.metric.compute().item(),
            "fw_IoU": self.metric_fw_iou.compute().item(),
            "precision": self.metric_precision.compute().item(),
            "recall": self.metric_recall.compute().item(),
            "f1": self.metric_f1.compute().item(),
        }

    @torch.no_grad()
    def _evaluate_gpu_cache(
        self,
        gpu_cache: GPUTensorCache,
        batch_size: int,
        *,
        collect_preds: bool = False,
        collect_confusions: bool = False,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor] | tuple[SegMetrics, torch.Tensor, torch.Tensor]":
        """Evaluate on a :class:`GPUTensorCache` and return segmentation metrics."""
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        pred_list: list[torch.Tensor] = []
        confusion_list: list[torch.Tensor] = []

        input_hw = (gpu_cache.masks.shape[-2], gpu_cache.masks.shape[-1])
        for features, masks in gpu_cache.ordered_batches(batch_size):
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model.head(features, *input_hw)
            for m in self._all_metrics:
                m.update(logits, masks)
            if collect_preds or collect_confusions:
                predictions = logits.argmax(dim=1)
                if collect_preds:
                    pred_list.append(predictions.cpu())
                if collect_confusions:
                    confusion_list.append(self._per_image_confusions(predictions, masks).cpu())

        metrics = self._compute_metrics()
        return self._evaluation_result(
            metrics,
            pred_list,
            confusion_list,
            collect_preds=collect_preds,
            collect_confusions=collect_confusions,
        )


def build_seg_probe_and_solver(
    model: nn.Module,
    num_classes: int,
    eval_cfg: "DictConfig",
    device: torch.device,
    lr: float,
) -> tuple[SegmentationProbe, SegmentationSolver]:
    """Build the frozen-backbone probe and its solver from the merged eval config."""
    from torchgeo_bench.config import instantiate

    layer_names = list(eval_cfg.segmentation.layers)
    if not layer_names:
        raise ValueError(
            "Segmentation evaluation requires eval.segmentation.layers to name "
            "spatial backbone layers. Refusing to probe the global backbone output."
        )
    probe = SegmentationProbe(
        backbone=model,
        layer_names=layer_names,
        num_classes=num_classes,
        head_type=eval_cfg.segmentation.head_type,
        freeze_backbone=True,
        temporal_pool=str(eval_cfg.segmentation.get("temporal_pool", "mean")),
    )
    criterion = instantiate(eval_cfg.segmentation.criterion)
    criterion_ignore = getattr(criterion, "ignore_index", None)
    solver = SegmentationSolver(
        model=probe,
        num_classes=num_classes,
        lr=lr,
        device=str(device),
        criterion=criterion,
        lr_scheduler=eval_cfg.segmentation.get("lr_scheduler", "cosine"),
        ignore_index=int(criterion_ignore) if criterion_ignore is not None else 255,
    )
    return probe, solver
