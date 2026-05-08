"""Model interface for torchgeo-bench.

Defines :class:`BenchModel`, the abstract base class that every benchmarkable
model inherits from.  The contract is split into two halves:

1. **Construction**: subclasses receive a
   :class:`list[~torchgeo_bench.datasets.base.BandSpec]` describing the input
   channels.  Per-channel mean/std/min/max statistics are available on each
   :class:`~torchgeo_bench.datasets.base.BandSpec` and are used by
   :meth:`normalize_inputs` to z-score the raw input tensor.

2. **Forward path**:

   * The public :meth:`forward_patch_features` is **sealed** — subclasses
     do not override it.  It always applies :meth:`normalize_inputs` before
     dispatching to :meth:`_forward_patch_features`, so normalization can't
     be silently forgotten.
   * Subclasses implement :meth:`_forward_patch_features` (the abstract
     hook) which receives the **already-normalized** ``(B, C, H, W)`` tensor
     and returns ``(B, K)`` embeddings.

Models whose backbones do their own normalization (e.g. OlmoEarth) override
:meth:`normalize_inputs` to identity.  Models that need a different
normalization strategy (ImageNet-style for pretrained RGB CNNs, weights-bound
``Normalize`` transforms for torchgeo wrappers, etc.) override
:meth:`normalize_inputs` with their own policy.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec


class BenchModel(nn.Module, ABC):
    """Abstract base interface for benchmarkable models.

    Args:
        bands: Ordered list of :class:`BandSpec` describing the input
            channels.  Length determines :attr:`num_channels`; the
            ``mean``/``std`` fields seed the default per-channel z-score
            applied by :meth:`normalize_inputs`.
    """

    def __init__(self, bands: list[BandSpec], **_: object) -> None:
        super().__init__()
        if not bands:
            raise ValueError("BenchModel requires a non-empty list of BandSpec.")
        self.bands: list[BandSpec] = list(bands)
        self.num_channels: int = len(self.bands)

        mean = torch.tensor([b.mean for b in self.bands], dtype=torch.float32).view(
            1, self.num_channels, 1, 1
        )
        std = torch.tensor([b.std for b in self.bands], dtype=torch.float32).view(
            1, self.num_channels, 1, 1
        )
        std = std.clamp_min(1e-8)
        self.register_buffer("input_mean", mean)
        self.register_buffer("input_std", std)

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Per-channel z-score against ``BandSpec.{mean, std}``.

        Buffers are cast to the input tensor's dtype on access so that the
        model works under mixed precision without surprises.

        Args:
            images: Raw input tensor of shape ``(B, C, H, W)``.

        Returns:
            Normalized tensor of the same shape and dtype as ``images``.
        """
        mean = self.input_mean.to(dtype=images.dtype)
        std = self.input_std.to(dtype=images.dtype)
        return (images - mean) / std

    @abstractmethod
    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Subclass hook — receives normalized ``(B, C, H, W)``, returns ``(B, K)``.

        Implementations should call only the backbone; the public
        :meth:`forward_patch_features` has already applied
        :meth:`normalize_inputs`.

        Args:
            images: Normalized input tensor of shape ``(B, C, H, W)``.
            bboxes: Optional bounding boxes, shape ``(B, 4)``.

        Returns:
            Embeddings tensor of shape ``(B, K)``.
        """
        raise NotImplementedError

    def forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a batch of vector embeddings ``(B, K)`` from raw inputs.

        Sealed: applies :meth:`normalize_inputs` then dispatches to
        :meth:`_forward_patch_features`.  Override
        :meth:`normalize_inputs` to change the normalization policy and
        :meth:`_forward_patch_features` to change the backbone forward.
        """
        return self._forward_patch_features(self.normalize_inputs(images), bboxes)

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Alias for :meth:`forward_patch_features`."""
        return self.forward_patch_features(images, bboxes)
