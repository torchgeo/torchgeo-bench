"""Segmentation Training Task Logic."""

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from .segmentation_probe import SegmentationProbe

logger = logging.getLogger(__name__)


class SegmentationBCELoss(nn.Module):
    """Binary cross-entropy loss with one-hot targets for segmentation.

    Applies per-class binary classification rather than multi-class CE.
    Often outperforms CrossEntropyLoss for frozen linear probes (Kerssies et al., 2024).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(masks, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return F.binary_cross_entropy_with_logits(logits, one_hot)


class SegmentationSolver:
    """A lightweight trainer for the SegmentationProbe."""

    # Common ignore index values used in segmentation datasets
    IGNORE_INDEX = 255

    def __init__(
        self,
        model: SegmentationProbe,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
        loss: str = "ce",
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
            criterion: Explicit loss module. If provided, overrides `loss`.
            loss: Loss type: "ce" (CrossEntropyLoss) or "bce" (binary CE over one-hot targets).
                  BCE consistently outperforms CE for frozen linear probes (Kerssies et al., 2024).
            lr_scheduler: LR schedule: "cosine" (CosineAnnealingLR) or "none" (constant LR).
            ignore_index: Label value to ignore in loss and metrics (default: 255).
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        self.lr_scheduler_type = lr_scheduler

        self.ignore_index = ignore_index
        # parameters can either be heads for linear probe or projectors + head for conv_block probe
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
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """Train the segmentation probe.

        Args:
            train_loader: Training data loader.
            val_loader: Optional validation data loader for per-epoch mIoU logging.
            epochs: Number of training epochs.
            verbose: Whether to show progress bars and epoch logs.
        """
        # Set up cosine LR schedule over the full training run
        if self.lr_scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        else:
            scheduler = None
            
        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not verbose)
            for _num_batches, batch in enumerate(pbar, start=1):
                if isinstance(batch, dict):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device).long()
                else:
                    images, masks = batch[0].to(self.device), batch[1].to(self.device).long()

                if masks.ndim == 4:
                    masks = masks.squeeze(1)

                self.optimizer.zero_grad()
                logits = self.model(images)

                loss = self.criterion(logits, masks)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if scheduler is not None:
                scheduler.step()

            if val_loader and verbose:
                miou = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch + 1} Val mIoU: {miou:.4f}")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate the model on a dataloader and return mean IoU.

        Args:
            dataloader: Evaluation data loader.

        Returns:
            Mean Intersection-over-Union (mIoU) score.
        """
        self.model.eval()
        self.metric.reset()

        self.metric.to(self.device)

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

            logits = self.model(images)

            self.metric.update(logits, masks)

        # Compute final score
        miou = self.metric.compute().item()
        return miou
