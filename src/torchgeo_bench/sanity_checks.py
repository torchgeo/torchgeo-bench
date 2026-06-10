"""Sanity checks for segmentation encoder evaluation."""

import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassJaccardIndex

from torchgeo_bench.models.segmentation_heads import ConvBlockHead, DPTHead, FPNHead, LinearHead
from torchgeo_bench.segmentation_probe import SegmentationProbe

logger = logging.getLogger(__name__)


def _build_fresh_head(probe: SegmentationProbe, num_classes: int, hidden_dim: int = 256) -> nn.Module:
    """Instantiate a new head of the same type as probe.head with fresh weights."""
    channels_list = probe.channels_list
    head_type = probe.head_type
    if head_type == "linear":
        return LinearHead(channels_list, num_classes)
    elif head_type == "conv_block":
        return ConvBlockHead(channels_list, num_classes, hidden_dim=hidden_dim)
    elif head_type == "fpn":
        return FPNHead(channels_list, num_classes, hidden_dim=hidden_dim)
    elif head_type == "dpt":
        return DPTHead(channels_list, num_classes, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown head_type: {head_type!r}")


def run_overfit_check(
    probe: SegmentationProbe,
    train_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    check_cfg: DictConfig,
    ignore_index: int = 255,
) -> dict:
    """Run the overfitting sanity check for a segmentation probe.

    Collects a small number of training batches, trains a fresh probe head on
    exactly those batches for a fixed number of gradient steps, then evaluates
    mIoU on the same batches. A functional encoder should easily memorize a
    handful of samples. Failure signals a broken backbone or misconfiguration.

    Args:
        probe: Built SegmentationProbe (backbone + hooks already registered).
        train_loader: Training DataLoader to collect batches from.
        num_classes: Number of segmentation classes.
        device: Target device.
        check_cfg: DictConfig with n_batches, overfit_steps, overfit_threshold, overfit_lr.
        ignore_index: Label value to ignore in loss and metrics.

    Returns:
        Dict with keys: passed, achieved_miou, threshold, n_batches, steps.
    """
    n_batches: int = int(check_cfg.get("overfit_n_batches", 2))
    steps: int = int(check_cfg.get("overfit_steps", 200))
    threshold: float = float(check_cfg.get("overfit_threshold", 0.95))
    lr: float = float(check_cfg.get("overfit_lr", 1e-3))
    hidden_dim: int = 256

    # --- Collect small subset of training data ---
    images_list: list[torch.Tensor] = []
    masks_list: list[torch.Tensor] = []
    for i, batch in enumerate(train_loader):
        if i >= n_batches:
            break
        if isinstance(batch, dict):
            imgs = batch["image"].to(device)
            msks = batch["mask"].to(device)
        else:
            imgs, msks = batch[0].to(device), batch[1].to(device)
        if msks.ndim == 4:
            msks = msks.squeeze(1)
        images_list.append(imgs)
        masks_list.append(msks.long())

    if not images_list:
        logger.warning("Overfit check: no batches available in train_loader — skipping.")
        return {
            "passed": False,
            "achieved_miou": 0.0,
            "threshold": threshold,
            "n_batches": 0,
            "steps": steps,
            "batch_size": 0,
            "unique_labels": 0,
            "feature_norm": 0.0,
            "feature_std": 0.0,
            "loss_delta": 0.0,
        }

    batch_size = images_list[0].shape[0]

    # --- Extract features using frozen backbone ---
    probe.backbone.eval()
    input_h, input_w = images_list[0].shape[-2:]
    use_amp = device.type == "cuda"

    layer_feature_batches: list[list[torch.Tensor]] = [[] for _ in probe.layer_names]
    with torch.no_grad():
        for imgs in images_list:
            probe._features.clear()
            with torch.autocast(device_type=device.type, enabled=use_amp):
                probe.backbone(imgs)
            for li, name in enumerate(probe.layer_names):
                feat = probe._process_feature(probe._features[name])
                layer_feature_batches[li].append(feat.detach())

    # Concatenate into (N, C, H, W) per layer
    stored_features: list[torch.Tensor] = [torch.cat(layer_feature_batches[li]) for li in range(len(probe.layer_names))]
    stored_masks: torch.Tensor = torch.cat(masks_list)
    n_samples = stored_masks.shape[0]

    unique_labels = int(stored_masks[stored_masks != ignore_index].unique().numel())
    feature_norm = float(torch.stack([f.norm(dim=1).mean() for f in stored_features]).mean().item())
    feature_std = float(torch.stack([f.std() for f in stored_features]).mean().item())

    # --- Build fresh head ---
    head = _build_fresh_head(probe, num_classes, hidden_dim=hidden_dim).to(device)
    head.train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Overfit training loop ---
    initial_loss: float | None = None
    final_loss: float = 0.0
    for _step in range(steps):
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = head(stored_features, input_h, input_w)
            loss = criterion(logits, stored_masks)
        if _step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    loss_delta = (initial_loss - final_loss) if initial_loss is not None else 0.0

    # --- Evaluate on the same data ---
    head.eval()
    metric = MulticlassJaccardIndex(
        num_classes=num_classes,
        ignore_index=ignore_index,
        average="macro",
    ).to(device)

    with torch.no_grad():
        batch_size = min(n_samples, 32)
        for start in range(0, n_samples, batch_size):
            s = slice(start, start + batch_size)
            feat_batch = [t[s] for t in stored_features]
            mask_batch = stored_masks[s]
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = head(feat_batch, input_h, input_w)
            metric.update(logits, mask_batch)

    achieved_miou = metric.compute().item()
    passed = achieved_miou >= threshold

    if not passed:
        logger.warning(
            f"Overfit check FAILED: achieved mIoU={achieved_miou:.3f} < threshold={threshold} "
            f"on {n_samples} samples ({n_batches} batches, {steps} steps). "
            "This may indicate broken layer hooks, degenerate features, or a head misconfiguration."
        )
    else:
        logger.info(f"Overfit check passed: mIoU={achieved_miou:.3f} >= {threshold}")

    return {
        "passed": passed,
        "achieved_miou": achieved_miou,
        "threshold": threshold,
        "n_batches": n_batches,
        "steps": steps,
        "batch_size": batch_size,
        "unique_labels": unique_labels,
        "feature_norm": feature_norm,
        "feature_std": feature_std,
        "initial_loss": initial_loss if initial_loss is not None else 0.0,
        "loss_delta": loss_delta,
    }



# for MODEL in \
#   sam3_encoder \
#   timm/convnext_base timm/convnext_large timm/convnext_large_dinov3 timm/convnext_small timm/convnext_tiny \
#   timm/densenet121 timm/densenet161 \
#   timm/efficientnet_b0 timm/efficientnet_b1 timm/efficientnet_b2 timm/efficientnet_b3 \
#   timm/maxvit_tiny_tf_224 \
#   timm/mobilenetv3_large_100 timm/mobilenetv3_small_100 \
#   timm/regnetx_002 timm/regnetx_008 timm/regnety_002 timm/regnety_008 \
#   timm/resnet18 timm/resnet34 timm/resnet50 timm/resnet101 \
#   timm/vgg16 timm/vgg19 \
#   torchgeo/dofa_base torchgeo/scalemae_large_fmow torchgeo/swinv2b_s2rgb_satlas_mi \
# ; do
#   echo "=== $MODEL ==="
#   torchgeo-bench overfit-check model=$MODEL dataset.names=[caffe,flair2] dataset.geobench_v2_root=/mnt/SSD2/nils/datasets/geobench2
# done