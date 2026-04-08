"""Segmentation Probe Module."""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

logger = logging.getLogger(__name__)


class SegmentationProbe(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        layer_names: list[str],
        num_classes: int,
        in_channels: int = 3,
        input_size: int | None = None,
        freeze_backbone: bool = True,
        head_type: str = "linear",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size
        self.head_type = head_type

        self.effective_classes = num_classes

        self._features: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

        found_layers = set()
        for name, module in self.backbone.named_modules():
            # remove starting "backbone." if present
            if name.startswith("backbone."):
                name = name.replace("backbone.", "", 1)
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
                found_layers.add(name)

        missing_layers = set(self.layer_names) - found_layers
        if missing_layers:
            logger.warning(f"The following layers were not found in the backbone: {missing_layers}")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.channels_list = self._dry_run_channels()

        if head_type == "linear":
            self.heads = nn.ModuleList()
            for c in self.channels_list:
                self.heads.append(
                    nn.Sequential(
                        nn.BatchNorm2d(c), nn.Conv2d(c, self.effective_classes, kernel_size=1)
                    )
                )

            if len(self.layer_names) > 1:
                self.scale_weights = nn.Parameter(torch.ones(len(self.layer_names)))

        elif head_type == "conv_block":
            embed_dim = hidden_dim or 256
            self.projectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
                        nn.BatchNorm2d(embed_dim),
                        nn.SiLU(inplace=True),
                    )
                    for c in self.channels_list
                ]
            )
            self.head = nn.Conv2d(
                embed_dim * len(self.hooks), self.effective_classes, kernel_size=1
            )

        # Metric
        self.miou_metric = MulticlassJaccardIndex(
            num_classes=self.effective_classes,
        )

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            self._features[name] = output

        return hook

    def _dry_run_channels(self) -> list[int]:
        device = next(self.backbone.parameters()).device
        dummy = torch.randn(1, 3, 224, 224, device=device)
        if not self.layer_names:
            self.layer_names = ["backbone_output"]
            self.hooks.append(self.backbone.register_forward_hook(self._hook_fn("backbone_output")))

        was_training = self.backbone.training
        self.backbone.eval()
        self._features.clear()
        with torch.no_grad():
            self.backbone(dummy)

        channels = []
        for name in self.layer_names:
            feat = self._features[name]
            if feat.ndim == 2:
                channels.append(feat.shape[1])
            elif feat.ndim == 3:
                channels.append(feat.shape[2])
            else:
                channels.append(feat.shape[1])
        self.backbone.train(was_training)
        return channels

    def _process_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 2:
            return feat.view(feat.shape[0], feat.shape[1], 1, 1)
        if feat.ndim == 3:
            B, L, C = feat.shape
            H = int(L**0.5)
            if H * H == L:
                return feat.permute(0, 2, 1).reshape(B, C, H, H)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_h, input_w = x.shape[-2:]

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                _ = self.backbone(x)
        else:
            _ = self.backbone(x)

        features = [self._process_feature(self._features[n]) for n in self.layer_names]

        if self.head_type == "linear":
            total_logits = 0
            for idx, (feat, head) in enumerate(zip(features, self.heads)):
                # get logits at low resolution to avoid OOM
                logits = head(feat)

                # upsample to Input Resolution
                if logits.shape[-2:] != (input_h, input_w):
                    logits = F.interpolate(
                        logits, size=(input_h, input_w), mode="bilinear", align_corners=False
                    )
                if len(self.layer_names) == 1:
                    return logits
                else:
                    total_logits = total_logits + (logits * self.scale_weights[idx])
            return total_logits

        else:
            # project extrated features to embed dim
            proj_feats = [proj(f) for f, proj in zip(features, self.projectors)]

            # find a common spatial feature size
            target_h, target_w = 0, 0
            for f in proj_feats:
                if f.shape[-2] > target_h:
                    target_h, target_w = f.shape[-2:]
            if target_h == 1:
                target_h, target_w = 16, 16

            # align features
            aligned_feats = []
            for f in proj_feats:
                if f.shape[-2:] != (target_h, target_w):
                    f = F.interpolate(
                        f, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )
                aligned_feats.append(f)

            x_fused = torch.cat(aligned_feats, dim=1)

            logits = self.head(x_fused)

            # upsample to target size
            if logits.shape[-2:] != (input_h, input_w):
                logits = F.interpolate(
                    logits, size=(input_h, input_w), mode="bilinear", align_corners=False
                )

            return logits
