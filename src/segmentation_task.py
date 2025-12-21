from .segmentation_probe import SegmentationProbe
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm

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
        self.optimizer = torch.optim.AdamW(
            self.model.head.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()

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

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", disable=not verbose)
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
                logger.info(f"Epoch {epoch+1} Val mIoU: {miou:.4f}")

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        conf_mat = torch.zeros((self.num_classes, self.num_classes), device=self.device)

        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device).long()
            else:
                images, masks = batch[0].to(self.device), batch[1].to(self.device).long()

            if masks.ndim == 4:
                masks = masks.squeeze(1)

            logits = self.model(images)
            preds = logits.argmax(dim=1)

            mask_vector = masks.flatten()
            pred_vector = preds.flatten()
            valid = (mask_vector >= 0) & (mask_vector < self.num_classes)

            idx = mask_vector[valid] * self.num_classes + pred_vector[valid]
            conf_mat += torch.bincount(idx, minlength=self.num_classes**2).reshape(
                self.num_classes, self.num_classes
            )

        intersection = torch.diag(conf_mat)
        union = conf_mat.sum(0) + conf_mat.sum(1) - intersection
        iou = intersection / (union + 1e-6)
        miou = iou.mean().item()
        return miou
