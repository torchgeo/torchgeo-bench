"""Uniform member predictor for the label-quality OOF substrate.

Every member is the *same* object regardless of backbone family: a
:class:`~torchgeo_bench.segmentation_probe.SegmentationProbe` with
``freeze_backbone=False`` (full fine-tuning) topped by a DPT/FPN decoder head,
trained on a fixed step budget via :class:`SegmentationSolver`. A member spec
just names a registry backbone, a head, and a seed — there is no CNN/FM branch.

``predict_proba`` returns softmax probabilities ``(N, C, H, W)`` at the *native
label resolution*: when a backbone runs at a coarser frame (patch-size
constraints) its logits are bilinearly upsampled back to the mask's ``(H, W)``
before softmax. Labels are never resampled — only the continuous prediction is.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..segmentation_probe import SegmentationProbe
from ..segmentation_task import SegmentationSolver

logger = logging.getLogger(__name__)


class Predictor:
    """A single trained OOF member: fixed-budget fine-tuning + native-frame softmax."""

    def __init__(
        self,
        probe: SegmentationProbe,
        solver: SegmentationSolver,
        max_steps: int | None,
        epoch_cap: int,
        device: str,
    ) -> None:
        self.probe = probe
        self.solver = solver
        self.max_steps = max_steps
        self.epoch_cap = epoch_cap
        self.device = device

    def fit(self, train_loader: DataLoader) -> "Predictor":
        """Fine-tune the member for the configured fixed step budget (no validation)."""
        self.solver.fit(
            train_loader,
            max_steps=self.max_steps,
            epoch_cap=self.epoch_cap,
            verbose=False,
        )
        return self

    @torch.no_grad()
    def predict_proba(self, loader: DataLoader) -> torch.Tensor:
        """Return ``(N, C, H, W)`` softmax at each batch's native label resolution."""
        self.probe.eval()
        chunks: list[torch.Tensor] = []
        for batch in loader:
            images, masks = _unpack(batch)
            target_hw = (masks.shape[-2], masks.shape[-1])
            logits = self.probe(images.to(self.device)).float()
            if logits.shape[-2:] != target_hw:
                logits = F.interpolate(
                    logits, size=target_hw, mode="bilinear", align_corners=False
                )
            chunks.append(logits.softmax(dim=1).cpu())
        return torch.cat(chunks, dim=0)


def build_member(spec: dict) -> Predictor:
    """Build a :class:`Predictor` from a member spec.

    Args:
        spec: Member configuration. Recognised keys:

            - ``backbone`` (required): an :class:`torch.nn.Module` or a zero-arg
              callable returning one (the resolved registry model).
            - ``layers``: spatial backbone layer names to hook.
            - ``num_classes``: number of segmentation classes.
            - ``head_type``: decoder head (``"fpn"``/``"dpt"``/...); default ``"fpn"``.
            - ``hidden_dim``: optional decoder hidden width.
            - ``seed``: RNG seed applied before head initialisation.
            - ``device``: ``"cpu"`` or ``"cuda"``.
            - ``criterion``: optional loss module.
            - ``ignore_index``: label value ignored in loss/metrics (default 255).
            - ``lr``/``weight_decay``/``backbone_lr``/``head_lr``: optimizer LRs.
            - ``max_steps``/``epoch_cap``: fixed training budget.

    Returns:
        An unfitted :class:`Predictor` wrapping an unfrozen probe and solver.
    """
    torch.manual_seed(spec.get("seed", 0))
    device = spec.get("device", "cpu")

    backbone = spec["backbone"]
    if callable(backbone) and not isinstance(backbone, nn.Module):
        backbone = backbone()

    num_classes = spec["num_classes"]
    probe = SegmentationProbe(
        backbone=backbone,
        layer_names=list(spec["layers"]),
        num_classes=num_classes,
        head_type=spec.get("head_type", "fpn"),
        hidden_dim=spec.get("hidden_dim"),
        freeze_backbone=False,
    )
    solver = SegmentationSolver(
        model=probe,
        num_classes=num_classes,
        lr=spec.get("lr", 1e-3),
        weight_decay=spec.get("weight_decay", 0.0),
        device=device,
        criterion=spec.get("criterion"),
        lr_scheduler=spec.get("lr_scheduler", "none"),
        ignore_index=spec.get("ignore_index", 255),
        backbone_lr=spec.get("backbone_lr"),
        head_lr=spec.get("head_lr"),
    )
    return Predictor(
        probe,
        solver,
        max_steps=spec.get("max_steps"),
        epoch_cap=spec.get("epoch_cap", 1000),
        device=device,
    )


def _unpack(batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a batch into ``(images, masks)`` collapsing a singleton mask channel."""
    if isinstance(batch, dict):
        images, masks = batch["image"], batch["mask"]
    else:
        images, masks = batch[0], batch[1]
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return images, masks
