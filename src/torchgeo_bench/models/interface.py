"""Model interface for torchgeo-bench.

This module defines a lightweight base class that geospatial / foundation models
can inherit from (or emulate) in order to be benchmarked with the
``torchgeo_bench.main`` module.

Contract (forward_features):
  Inputs:
    images: torch.Tensor shape (B, C, H, W) with float32 values in [0, 1] (or
            already normalized if the dataset transform performs normalization).
    bboxes: Optional torch.Tensor shape (B, 4) with (minx, miny, maxx, maxy)
            coordinates in EPSG:4326. For pure image models this can be None.
  Output:
    embeddings: torch.Tensor shape (B, K) where K is the embedding dimension.

To integrate an existing timm / torchgeo model you can create a thin wrapper
class implementing ``forward_features`` (and delegating any internal feature
extraction utilities) while leaving the original model untouched.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BenchModel(nn.Module, ABC):
    """Abstract base interface for benchmarkable models."""

    def __init__(self, num_channels: int, *_, **__):  # type: ignore[no-untyped-def]
        """Initialize BenchModel.

        Args:
            num_channels: Number of input image channels.
        """
        super().__init__()
        self.num_channels = num_channels

    @abstractmethod
    def forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch of vector embeddings (B, K).

        Args:
            images: Input images, shape (B, C, H, W).
            bboxes: Optional bounding boxes, shape (B, 4).

        Returns:
            Embeddings tensor of shape (B, K).
        """
        raise NotImplementedError

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch of vector embeddings (B, K).

        Alias for ``forward_patch_features``.

        Args:
            images: Input images, shape (B, C, H, W).
            bboxes: Optional bounding boxes, shape (B, 4).

        Returns:
            Embeddings tensor of shape (B, K).
        """
        return self.forward_patch_features(images, bboxes)
