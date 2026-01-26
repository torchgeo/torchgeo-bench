"""Segmentation Training Task Logic."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from .segmentation_probe import SegmentationProbe

logger = logging.getLogger(__name__)


class SegmentationSolver:
    """A lightweight trainer for the SegmentationProbe."""

    def __init__(
        self,
        model: SegmentationProbe,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cuda",
        criterion: Optional[nn.Module] = None,
    ) -> None:
        """Initialize the SegmentationSolver.

        Args:
            model: The SegmentationProbe model to train.
            num_classes: Number of segmentation classes.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the optimizer.
            device: Device to run training on ('cuda' or 'cpu').
            criterion: Loss function to use. If None, defaults to CrossEntropyLoss.
        """
        self.model = model.to(device)
        self.num_classes = num_classes
        self.device = device
        # parameters can either be heads for linear probe or projectors + head for conv_block probe
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

        self.metric = MulticlassJaccardIndex(
            num_classes=self.num_classes,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 5,
        verbose: bool = True,
    ) -> None:
        for epoch in range(epochs):
            self.model.train()
            if self.model.freeze_backbone:
                self.model.backbone.eval()

            total_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", disable=not verbose)
            for batch in pbar:
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
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if val_loader and verbose:
                miou = self.evaluate(val_loader)
                logger.info(f"Epoch {epoch + 1} Val mIoU: {miou:.4f}")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
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
