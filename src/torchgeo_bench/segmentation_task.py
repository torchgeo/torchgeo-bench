"""Segmentation Training Task Logic."""

import logging
import math
from collections.abc import Callable

import torch
import torch.nn as nn
from rich.progress import track
from torch.utils.data import DataLoader
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

# Scalar aggregates plus ``per_class_IoU``, which is a ``list[float]`` of length
# ``num_classes``. Consumers writing flat rows must select scalars by key.
SegMetrics = dict[str, float | list[float]]


class NonFiniteLossError(RuntimeError):
    """Raised when training produces a NaN or infinite loss.

    Distinct from a generic ``RuntimeError`` so callers can tell divergence
    apart from real failures (OOM, shape bugs) and react differently — the
    hparam search turns this into ``optuna.TrialPruned`` rather than letting
    the trial fail.
    """


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
        backbone_lr: float | None = None,
        head_lr: float | None = None,
        max_grad_norm: float | None = 1.0,
    ) -> None:
        """Initialize the SegmentationSolver.

        Args:
            model: The SegmentationProbe model to train.
            num_classes: Number of segmentation classes.
            lr: Learning rate for the optimizer. Ignored when both
                ``backbone_lr`` and ``head_lr`` are given.
            weight_decay: Weight decay for the optimizer.
            device: Device to run training on ('cuda' or 'cpu').
            criterion: Loss module. Defaults to CrossEntropyLoss with ignore_index.
            lr_scheduler: LR schedule: "cosine" (CosineAnnealingLR) or "none" (constant LR).
            ignore_index: Label value to ignore in loss and metrics (default: 255).
            backbone_lr: Learning rate for the backbone param group. When set
                together with ``head_lr``, the optimizer is built with two
                differential-LR param groups (backbone / head) instead of a
                single ``lr`` group.
            head_lr: Learning rate for the head param group; see ``backbone_lr``.
            max_grad_norm: Global gradient-norm clip applied in the full
                fine-tuning path (:meth:`_train_on_batch`). ``None`` disables
                clipping. Defaults to 1.0 — unclipped AMP fp16 training let a
                single non-finite gradient poison AdamW's ``exp_avg`` /
                ``exp_avg_sq`` permanently, which is what drove the mid-training
                divergence to NaN in the ViT segmentation runs.
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler
        self.val_history: list[float] = []
        # Per-evaluation training curve, populated only by the instrumented
        # fixed-budget path (``fit(max_steps=..., val_loader=..., eval_every=...)``).
        self.history: list[dict] = []

        self.ignore_index = ignore_index
        self.max_grad_norm = max_grad_norm
        self.optimizer = self._build_optimizer(lr, weight_decay, backbone_lr, head_lr)

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
        # Per-class IoU: free on eval passes already being paid for, and what
        # separates "hard dataset" from "broken dataset" — a low mIoU driven by
        # two dead classes reads very differently from a uniformly low one.
        self.metric_per_class_iou = MulticlassJaccardIndex(
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
            average=None,
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
            self.metric_per_class_iou,
            self.metric_precision,
            self.metric_recall,
            self.metric_f1,
        ]

        self.use_amp = device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.device_type = torch.device(device).type

    def _build_optimizer(
        self,
        lr: float,
        weight_decay: float,
        backbone_lr: float | None,
        head_lr: float | None,
    ) -> torch.optim.Optimizer:
        """Build the AdamW optimizer.

        With both ``backbone_lr`` and ``head_lr`` set, two differential-LR param
        groups (backbone / head) are created from the trainable parameters;
        otherwise a single ``lr`` group over all trainable parameters is used.
        """
        if backbone_lr is not None and head_lr is not None:
            backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
            head_params = [p for p in self.model.head.parameters() if p.requires_grad]
            param_groups = [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ]
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

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
        max_steps: int | None = None,
        epoch_cap: int = 1000,
        eval_every: int | None = None,
        step_callback: "Callable[[int, float], bool] | None" = None,
    ) -> float | None:
        """Train the segmentation probe.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader for per-epoch mIoU logging.
                In the fixed-budget regime it is used only when ``eval_every``
                is also set (see below).
            epochs: Number of training epochs (epoch-based regime).
            verbose: Whether to show progress bars and epoch logs.
            max_steps: If set, train for exactly this many optimizer steps
                (fixed-budget regime), cycling through the loader across up to
                ``epoch_cap`` epochs, with no scheduling or early-stopping.
                Overrides ``epochs``.
            epoch_cap: Maximum number of loader passes when ``max_steps`` is set.
            eval_every: Fixed-budget regime only. When set together with
                ``val_loader``, evaluate on ``val_loader`` every this many
                optimizer steps (and once at the final step), recording each
                result in :attr:`history`. Defaults to ``None`` — no mid-training
                evaluation, so the fixed-budget path behaves exactly as before.
            step_callback: Fixed-budget regime only. Called as
                ``step_callback(step, val_mIoU)`` after every mid-training
                evaluation; returning ``True`` stops training early. Used by the
                hyperparameter search to feed an Optuna pruner. Note that this is
                the *only* way the fixed-budget loop can terminate early — the
                training regime itself never early-stops (see D9).

        Returns:
            Val mIoU from the final epoch if val_loader is given (epoch-based
            regime), the last recorded val mIoU in the fixed-budget regime when
            ``eval_every`` is set, else None.
        """
        if max_steps is not None:
            return self._fit_fixed_budget(
                train_loader,
                max_steps,
                epoch_cap,
                verbose,
                val_loader=val_loader,
                eval_every=eval_every,
                step_callback=step_callback,
            )

        scheduler = self._make_scheduler(epochs)
        last_val_miou: float | None = None
        self.val_history = []

        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            total_loss = 0.0

            desc = f"Epoch {epoch + 1}/{epochs}"
            batches = track(train_loader, description=desc) if verbose else train_loader
            for batch in batches:
                total_loss += self._train_on_batch(batch)

            if scheduler is not None:
                scheduler.step()

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                last_val_miou = val_metrics["mIoU"]
                self.val_history.append(last_val_miou)
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch + 1, last_val_miou)

        return last_val_miou

    def _train_on_batch(self, batch) -> float:
        """Run one optimizer step on a single batch and return its loss value.

        Raises:
            NonFiniteLossError: If the loss is NaN or infinite. Raised at the
                *first* occurrence rather than at the next eval boundary, so the
                caller sees the step that actually broke.
        """
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

        # Catch divergence before the bad value reaches the optimizer: once a
        # non-finite gradient lands in AdamW's exp_avg/exp_avg_sq, every
        # subsequent step is poisoned and the run never recovers.
        if not torch.isfinite(loss):
            raise NonFiniteLossError(f"Non-finite training loss: {loss.item()}")

        self.scaler.scale(loss).backward()

        if self.max_grad_norm is not None:
            # Must come after unscale_: clipping still-scaled gradients would
            # threshold against the scaler's current scale factor, i.e. at an
            # effectively random magnitude that changes whenever the scaler
            # backs off.
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                (p for p in self.model.parameters() if p.requires_grad),
                max_norm=self.max_grad_norm,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item()

    def _fit_fixed_budget(
        self,
        train_loader: DataLoader,
        max_steps: int,
        epoch_cap: int,
        verbose: bool,
        *,
        val_loader: DataLoader | None = None,
        eval_every: int | None = None,
        step_callback: "Callable[[int, float], bool] | None" = None,
    ) -> float | None:
        """Train for exactly ``max_steps`` optimizer steps.

        Cycles through ``train_loader`` across up to ``epoch_cap`` passes,
        stopping the moment ``max_steps`` steps have run. Learning rate is held
        constant (no scheduler) and the training regime never early-stops.

        Instrumentation is opt-in: with ``val_loader`` and ``eval_every`` both
        set, the held-out set is evaluated every ``eval_every`` steps and at the
        final step, and each result is appended to :attr:`history` alongside the
        mean train loss since the previous evaluation. With either omitted the
        loop is byte-identical to its uninstrumented behavior, so the
        label-quality launch pays nothing for the search's needs.

        Returns:
            The last recorded val mIoU when evaluating, else ``None``.
        """
        self.val_history = []
        self.history = []
        instrument = val_loader is not None and eval_every is not None and eval_every > 0

        steps = 0
        loss_sum = 0.0
        loss_count = 0
        last_val_miou: float | None = None

        for _ in range(epoch_cap):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            desc = f"Step {steps}/{max_steps}"
            batches = track(train_loader, description=desc) if verbose else train_loader
            for batch in batches:
                loss = self._train_on_batch(batch)
                steps += 1
                loss_sum += loss
                loss_count += 1

                final = steps >= max_steps
                if instrument and (final or steps % eval_every == 0):
                    train_loss = loss_sum / loss_count if loss_count else float("nan")
                    loss_sum, loss_count = 0.0, 0
                    metrics = self.evaluate(val_loader)
                    last_val_miou = metrics["mIoU"]
                    self.val_history.append(last_val_miou)
                    self.history.append(
                        {
                            "step": steps,
                            "train_loss": train_loss,
                            "val_mIoU": last_val_miou,
                            "per_class_IoU": list(metrics["per_class_IoU"]),
                        }
                    )
                    if verbose:
                        logger.info(
                            "Step %d train_loss=%.17g val mIoU=%.17g",
                            steps,
                            train_loss,
                            last_val_miou,
                        )
                    # ``evaluate`` leaves the model in eval mode; the fixed-budget
                    # loop only calls ``train()`` once per loader pass, so without
                    # this the remainder of the pass would train with eval-mode
                    # BatchNorm/dropout.
                    self.model.train()
                    if self.model.freeze_backbone:
                        self.model.backbone.eval()
                    if step_callback is not None and step_callback(steps, last_val_miou):
                        return last_val_miou

                if final:
                    return last_val_miou
        return last_val_miou

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        collect_preds: bool = False,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor]":
        """Evaluate the model on a dataloader and return segmentation metrics.

        Args:
            dataloader: Evaluation data loader.
            collect_preds: If True, also return predicted class maps (N, H, W) int64.

        Returns:
            Dict of metric name → value, or (metrics_dict, preds_tensor) when
            collect_preds=True.
        """
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        pred_list: list[torch.Tensor] = []

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

        metrics = self._compute_metrics()
        if collect_preds:
            return metrics, torch.cat(pred_list, dim=0)
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
        self.val_history = []
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

                # Same divergence guard as ``_train_on_batch``: a non-finite loss
                # here would otherwise reach AdamW and poison exp_avg/exp_avg_sq
                # for every subsequent step.
                if not torch.isfinite(loss):
                    raise NonFiniteLossError(f"Non-finite training loss: {loss.item()}")

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            if scheduler is not None:
                scheduler.step()

            if gpu_val is not None:
                val_metrics = self._evaluate_gpu_cache(gpu_val, batch_size)
                last_val_miou = val_metrics["mIoU"]
                self.val_history.append(last_val_miou)
                if verbose:
                    logger.info("Epoch %d Val mIoU: %.17g", epoch + 1, last_val_miou)

        return last_val_miou

    def evaluate_cached(
        self,
        cache: CachedFeaturesDataset,
        batch_size: int = 64,
        collect_preds: bool = False,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor]":
        """Evaluate on a CachedFeaturesDataset.

        The cache is moved to GPU as a :class:`GPUTensorCache` for zero
        per-batch host→device transfers.

        Args:
            cache: Pre-extracted features (output of
                :meth:`SegmentationProbe.extract_segmentation_features`).
            batch_size: Batch size for iterating over the cache.
            collect_preds: If True, also return predicted class maps (N, H, W) int64.

        Returns:
            Dict of metric name → value, or (metrics_dict, preds_tensor) when
            collect_preds=True.
        """
        gpu_cache = GPUTensorCache.from_cached(cache, self.device)
        return self._evaluate_gpu_cache(gpu_cache, batch_size, collect_preds=collect_preds)

    def _compute_metrics(self) -> "SegMetrics":
        """Compute and return all metrics as a dict.

        ``per_class_IoU`` is a length-``num_classes`` list (not a scalar), so
        consumers writing flat metric rows must select it out explicitly.
        """
        return {
            "mIoU": self.metric.compute().item(),
            "fw_IoU": self.metric_fw_iou.compute().item(),
            "per_class_IoU": self.metric_per_class_iou.compute().tolist(),
            "precision": self.metric_precision.compute().item(),
            "recall": self.metric_recall.compute().item(),
            "f1": self.metric_f1.compute().item(),
        }

    @torch.no_grad()
    def _evaluate_gpu_cache(
        self,
        gpu_cache: GPUTensorCache,
        batch_size: int,
        collect_preds: bool = False,
    ) -> "SegMetrics | tuple[SegMetrics, torch.Tensor]":
        """Evaluate on a :class:`GPUTensorCache` and return segmentation metrics."""
        self.model.eval()
        for m in self._all_metrics:
            m.reset()
            m.to(self.device)

        pred_list: list[torch.Tensor] = []

        input_hw = (gpu_cache.masks.shape[-2], gpu_cache.masks.shape[-1])
        for features, masks in gpu_cache.ordered_batches(batch_size):
            with torch.autocast(device_type=self.device_type, enabled=self.use_amp):
                logits = self.model.head(features, *input_hw)
            for m in self._all_metrics:
                m.update(logits, masks)
            if collect_preds:
                pred_list.append(logits.argmax(dim=1).cpu())

        metrics = self._compute_metrics()
        if collect_preds:
            return metrics, torch.cat(pred_list, dim=0)
        return metrics
