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

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy, build_normalizer


class BenchModel(nn.Module, ABC):
    """Abstract base interface for benchmarkable models.

    Args:
        bands: Ordered list of :class:`BandSpec` describing the input
            channels.  Length determines :attr:`num_channels`.
        normalization: Selectable input-normalisation strategy.  See
            :class:`~torchgeo_bench.models._normalization.NormalizationStrategy`.
            Defaults to ``"bandspec_zscore"``.

    Subclasses may declare:

    * :attr:`expected_input_unit` — what scale the pretrained backbone
      was fed at training (e.g. ``s2_dn``, ``reflectance_0_1``,
      ``uint8``).  Used by the ``model_native`` strategy.
    * :attr:`pretrain_mean` / :attr:`pretrain_std` — per-channel
      normalisation applied *after* unit conversion under
      ``model_native``.
    """

    expected_input_unit: InputUnit | None = None
    pretrain_mean: list[float] | None = None
    pretrain_std: list[float] | None = None

    def __init__(
        self,
        bands: list[BandSpec],
        normalization: NormalizationStrategy | str = NormalizationStrategy.BANDSPEC_ZSCORE,
        **_: object,
    ) -> None:
        super().__init__()
        if not bands:
            raise ValueError("BenchModel requires a non-empty list of BandSpec.")
        self.bands: list[BandSpec] = list(bands)
        self.num_channels: int = len(self.bands)
        self.normalization = NormalizationStrategy(normalization)
        self._normalizer = build_normalizer(
            self.normalization,
            bands=self.bands,
            expected_input_unit=self.expected_input_unit,
            pretrain_mean=self.pretrain_mean,
            pretrain_std=self.pretrain_std,
        )

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalisation strategy."""
        return self._normalizer(images)

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
