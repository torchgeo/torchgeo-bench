import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentationProbe(nn.Module):
    """A generic segmentation probe that wraps a backbone model.

    It extracts feature maps from specified layers via hooks, upsamples them
    to the input resolution, concatenates them, and applies a simple Conv block
    (linear probe or MLP) to predict segmentation masks.
    """

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
        """Initialize the SegmentationProbe.

        Args:
            backbone: The feature extracting base model to probe (e.g. smptimm model).
            layer_names: List of layer names (strings) to extract features from.
                        Use `dict(model.named_modules()).keys()` to find names
            num_classes: Number of segmentation classes.
            in_channels: Number of input channels (e.g. 3 for RGB).
            input_size: Optional input image size (H=W). If None, defaults to 224.
            freeze_backbone: Whether to freeze backbone parameters, default True.
            head_type: 'linear' (1x1 conv) or 'mlp' (Conv3x3->BN->ReLU->Conv1x1).
            hidden_dim: Hidden dimension for 'mlp' head type. If None, defaults to
                        max(feature_channels // 2, num_classes * 2).
        """
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self.freeze_backbone = freeze_backbone
        self.in_channels = in_channels
        self.input_size = input_size
        self.num_classes = num_classes
        self._features: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []

        # register hooks for layers from which we want to extract features
        found_layers = 0
        for name, module in self.backbone.named_modules():
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
                found_layers += 1

        if found_layers != len(self.layer_names):
            logger.warning(
                f"Requested {len(self.layer_names)} layers but only found {found_layers}. "
                f"Check layer names: {self.layer_names}"
            )

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        feature_channels = self._dry_run_channels()

        if head_type == "linear":
            self.head = nn.Conv2d(feature_channels, num_classes, kernel_size=1)
        elif head_type == "conv_block":
            dim = hidden_dim or max(feature_channels // 2, num_classes * 2)
            self.head = nn.Sequential(
                nn.Conv2d(feature_channels, dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(dim, num_classes, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def _hook_fn(self, name: str):
        """Store feature maps from specified layers.

        Args:
            name: Name of the layer to hook.
        """

        def hook(module, input, output):
            self._features[name] = output

        return hook

    def _dry_run_channels(self) -> int:
        """Run a dummy input to calculate total concatenated channel count."""
        device = next(self.backbone.parameters()).device
        if self.input_size is not None:
            dummy = torch.randn(
                1, self.in_channels, self.input_size, self.input_size, device=device
            )
        else:
            dummy = torch.randn(1, self.in_channels, 224, 224, device=device)

        # set to eval to avoid batchnorm updates during dry run
        training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            try:
                _ = self.backbone(dummy)
            except Exception:
                raise RuntimeError(
                    "Backbone model failed during dry run. "
                    f"Check if the model can process a dummy input of shape {(1, self.in_channels, self.input_size or 224, self.input_size or 224)}, otherwise specify input_size correctly."
                )

        total = 0
        for name in self.layer_names:
            if name in self._features:
                total += self._features[name].shape[1]
            else:
                raise RuntimeError(f"Layer {name} not found during dry run.")

        self.backbone.train(training)
        return total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the segmentation probet that returns segmentation logits.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Segmentation logits of shape (B, num_classes, H, W)
        """
        input_h, input_w = x.shape[-2], x.shape[-1]

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                _ = self.backbone(x)
        else:
            _ = self.backbone(x)

        features_list = []
        for name in self.layer_names:
            feat = self._features.get(name)
            if feat is None:
                continue

            if feat.shape[-2:] != (input_h, input_w):
                feat = F.interpolate(
                    feat, size=(input_h, input_w), mode="bilinear", align_corners=False
                )
            features_list.append(feat)

        x_cat = torch.cat(features_list, dim=1)

        return self.head(x_cat)
