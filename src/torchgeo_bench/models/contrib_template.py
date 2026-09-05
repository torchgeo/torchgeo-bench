"""Copy-paste skeleton for external model contributions.

This file is **not** part of the public ``torchgeo_bench.models`` namespace.
Copy it to your working directory, fill in the TODO sections, and follow the
contribution guide at ``docs/user/eval_own_model.rst`` (Stage 1) or
``docs/user/contribute_model.rst`` (Stage 2).
"""

import logging

import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

logger = logging.getLogger(__name__)


class NewModel(BenchModel):
    """Template BenchModel subclass — fill in the TODOs before running.

    The runner calls ``NewModel(bands=<list[BandSpec]>, **yaml_kwargs)`` once
    per dataset.  ``self.num_channels`` is set to ``len(bands)`` by
    ``BenchModel.__init__`` before your code runs.

    Args:
        bands: Ordered list of :class:`BandSpec` from the dataset wrapper.
            Do **not** include ``bands`` in the Hydra YAML — the runner
            injects it at construction time.
        pretrained: Load pretrained weights (default: ``True``).
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        pretrained: bool = True,
        # TODO: add any extra kwargs your backbone needs and mirror them in
        #       src/torchgeo_bench/conf/model/<name>.yaml
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, normalization="bandspec_zscore")
        # Use identity normalization if the backbone handles raw inputs internally.
        self.backbone = nn.Identity()  # TODO: replace with your backbone
        logger.info(
            "NewModel initialized with %d input channels (pretrained=%s)",
            self.num_channels,
            pretrained,
        )

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
        _bboxes: torch.Tensor | None = None,  # required by interface; ignore
    ) -> torch.Tensor:
        """Return embeddings ``(B, K)`` from already-normalized inputs.

        ``images`` has shape ``(B, C, H, W)`` and has already been passed
        through ``normalize_inputs`` by the sealed ``forward_patch_features``.
        If you chose ``normalization="identity"`` above, ``images`` is the
        raw sensor tensor.

        Args:
            images: Normalized input tensor of shape ``(B, C, H, W)``.

        Returns:
            Embedding tensor of shape ``(B, K)``.
        """
        x = self.backbone(images)
        if x.ndim == 4:  # (B, K, H, W) — pool spatial dims
            x = x.flatten(start_dim=2).mean(dim=-1)
        return x  # (B, K)
