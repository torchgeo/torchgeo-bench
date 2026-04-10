"""Segmentation Probe Module."""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex

logger = logging.getLogger(__name__)


class SegmentationProbe(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        layer_names: List[str],
        num_classes: int,
        in_channels: int = 3,
        input_size: Optional[int] = None,
        freeze_backbone: bool = True,
        head_type: str = "linear",
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size
        self.head_type = head_type

        self.effective_classes = num_classes

        self._features: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

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

        elif head_type == "fpn":
            self._build_fpn_head(hidden_dim or 256)

        elif head_type == "aspp":
            self._build_aspp_head(hidden_dim or 256)

        else:
            raise ValueError(f"Unknown head_type: {head_type!r}. Choose from: linear, conv_block, fpn, aspp")

        # Metric
        self.miou_metric = MulticlassJaccardIndex(
            num_classes=self.effective_classes,
        )

    # ------------------------------------------------------------------
    # Head builders
    # ------------------------------------------------------------------

    def _build_fpn_head(self, hidden_dim: int) -> None:
        """Build an FPN-style decoder head.

        Layers must be provided in coarse-to-fine order (deepest/lowest-resolution first).
        E.g. for ResNet: ["layer4", "layer3", "layer2", "layer1"]

        Each stage gets:
          - A lateral 1×1 conv to project to hidden_dim
          - A 3×3 refinement conv (BN + ReLU)

        The top-down pathway adds upsampled coarser features to finer ones.
        All levels are upsampled to the finest level, concatenated, then
        projected to num_classes with a 1×1 conv.
        """
        self.fpn_hidden_dim = hidden_dim
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False) for c in self.channels_list]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in self.channels_list
            ]
        )
        self.fpn_head = nn.Conv2d(hidden_dim * len(self.channels_list), self.effective_classes, kernel_size=1)

    def _build_aspp_head(self, hidden_dim: int) -> None:
        """Build an ASPP (Atrous Spatial Pyramid Pooling) head.

        Uses the deepest layer only (last entry in layer_names).
        Parallel branches: 1×1 conv, dilated 3×3 (r=6,12,18), global avg pool.
        Outputs are concatenated, projected, then classified.
        """
        in_ch = self.channels_list[-1]  # deepest layer only
        self.aspp_hidden_dim = hidden_dim

        self.aspp_branches = nn.ModuleList([
            # 1×1 conv
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            # dilated 3×3, r=6
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            # dilated 3×3, r=12
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
            # dilated 3×3, r=18
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=18, dilation=18, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ),
        ])
        # Global average pooling branch
        self.aspp_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Projection after concat (5 branches × hidden_dim)
        self.aspp_proj = nn.Sequential(
            nn.Conv2d(hidden_dim * 5, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.aspp_head = nn.Conv2d(hidden_dim, self.effective_classes, kernel_size=1)

    # ------------------------------------------------------------------
    # Hook / dry-run helpers
    # ------------------------------------------------------------------

    def _hook_fn(self, name: str):
        def hook(module, input, output):
            self._features[name] = output

        return hook

    def _dry_run_channels(self) -> List[int]:
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

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

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
            return self._forward_linear(features, input_h, input_w)
        elif self.head_type == "conv_block":
            return self._forward_conv_block(features, input_h, input_w)
        elif self.head_type == "fpn":
            return self._forward_fpn(features, input_h, input_w)
        else:  # aspp
            return self._forward_aspp(features, input_h, input_w)

    def _forward_linear(
        self, features: list[torch.Tensor], input_h: int, input_w: int
    ) -> torch.Tensor:
        total_logits = 0
        for idx, (feat, head) in enumerate(zip(features, self.heads)):
            logits = head(feat)
            if logits.shape[-2:] != (input_h, input_w):
                logits = F.interpolate(
                    logits, size=(input_h, input_w), mode="bilinear", align_corners=False
                )
            if len(self.layer_names) == 1:
                return logits
            else:
                total_logits = total_logits + (logits * self.scale_weights[idx])
        return total_logits

    def _forward_conv_block(
        self, features: list[torch.Tensor], input_h: int, input_w: int
    ) -> torch.Tensor:
        proj_feats = [proj(f) for f, proj in zip(features, self.projectors)]

        target_h, target_w = 0, 0
        for f in proj_feats:
            if f.shape[-2] > target_h:
                target_h, target_w = f.shape[-2:]
        if target_h == 1:
            target_h, target_w = 16, 16

        aligned_feats = []
        for f in proj_feats:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(
                    f, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            aligned_feats.append(f)

        x_fused = torch.cat(aligned_feats, dim=1)
        logits = self.head(x_fused)

        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits

    def _forward_fpn(
        self, features: list[torch.Tensor], input_h: int, input_w: int
    ) -> torch.Tensor:
        """Top-down FPN forward pass.

        features: coarse-to-fine order (deepest first).
        Produces top-down merged feature maps, all upsampled to the finest
        resolution, concatenated, then classified.
        """
        # Apply lateral projections
        laterals = [lat(f) for f, lat in zip(features, self.laterals)]

        # Top-down merging: from coarsest (index 0) to finest (index -1)
        for i in range(len(laterals) - 1):
            target_size = laterals[i + 1].shape[-2:]
            laterals[i + 1] = laterals[i + 1] + F.interpolate(
                laterals[i], size=target_size, mode="bilinear", align_corners=False
            )

        # Apply refinement convs
        fpn_outs = [conv(p) for p, conv in zip(laterals, self.fpn_convs)]

        # Upsample all to finest resolution (last element = finest)
        finest_size = fpn_outs[-1].shape[-2:]
        aligned = []
        for f in fpn_outs:
            if f.shape[-2:] != finest_size:
                f = F.interpolate(f, size=finest_size, mode="bilinear", align_corners=False)
            aligned.append(f)

        fused = torch.cat(aligned, dim=1)
        logits = self.fpn_head(fused)

        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits

    def _forward_aspp(
        self, features: list[torch.Tensor], input_h: int, input_w: int
    ) -> torch.Tensor:
        """ASPP forward pass using the deepest feature map only."""
        feat = features[-1]  # deepest layer

        # If feature is spatially degenerate (1×1), fall back to nearest-sized map
        if feat.shape[-1] == 1:
            feat = F.interpolate(feat, size=(14, 14), mode="bilinear", align_corners=False)

        branch_outs = [branch(feat) for branch in self.aspp_branches]

        # Global pooling branch: upsample back to feat spatial size
        gap_out = self.aspp_gap(feat)
        gap_out = F.interpolate(gap_out, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        branch_outs.append(gap_out)

        fused = torch.cat(branch_outs, dim=1)
        fused = self.aspp_proj(fused)
        logits = self.aspp_head(fused)

        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits
