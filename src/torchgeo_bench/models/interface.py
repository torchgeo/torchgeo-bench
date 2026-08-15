"""Model interface for torchgeo-bench.

Defines :class:`BenchModel`; see its docstring for the construction and
sealed-forward contract.
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
        normalization: Input-normalisation strategy name (one of
            ``bandspec_zscore`` / ``model_native`` / ``minmax`` /
            ``minmax_zscore`` / ``identity``).  Defaults to
            ``"bandspec_zscore"``.

    Subclasses may declare:

    * ``expected_input_unit`` — what scale the pretrained backbone was
      fed at training (e.g. ``s2_dn``, ``reflectance_0_1``, ``uint8``).
      Used by the ``model_native`` strategy.
    * ``pretrain_mean`` / ``pretrain_std`` — per-channel normalisation
      applied *after* unit conversion under ``model_native``.
    """

    expected_input_unit: InputUnit | None = None
    pretrain_mean: list[float] | None = None
    pretrain_std: list[float] | None = None

    #: Set by wrappers whose backbone normalises internally (OlmoEarth runs its
    #: own per-modality ``Normalizer``).  Their ``normalize_inputs`` is already
    #: the model-native pipeline, so the strategy normaliser is never used and
    #: must not be validated against ``expected_input_unit``.
    handles_own_normalization: bool = False

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
        if self.handles_own_normalization:
            self._normalizer = lambda x: x
            return
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
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Subclass hook — receives normalized ``(B, C, H, W)``, returns ``(B, K)``.

        Implementations should call only the backbone; the public
        :meth:`forward_patch_features` has already applied
        :meth:`normalize_inputs`.

        Args:
            images: Normalized input tensor of shape ``(B, C, H, W)``.

        Returns:
            Embeddings tensor of shape ``(B, K)``.
        """
        raise NotImplementedError

    def forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Return a batch of vector embeddings ``(B, K)`` from raw inputs.

        Sealed: applies :meth:`normalize_inputs` then dispatches to
        :meth:`_forward_patch_features`.  Override
        :meth:`normalize_inputs` to change the normalization policy and
        :meth:`_forward_patch_features` to change the backbone forward.
        """
        return self._forward_patch_features(self.normalize_inputs(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`forward_patch_features`."""
        return self.forward_patch_features(images)
