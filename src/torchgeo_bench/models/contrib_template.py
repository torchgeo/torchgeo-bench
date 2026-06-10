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


class MyGeoFM(BenchModel):
    """Template BenchModel subclass — fill in the TODOs before running.

    The runner calls ``MyGeoFM(bands=<list[BandSpec]>, **yaml_kwargs)`` once
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
        # --- Choose your normalization strategy ---
        #
        # Option A — "bandspec_zscore" (default, safe choice):
        #   Framework z-scores each channel using dataset BandSpec mean/std.
        #   Use for most backbones trained on normalized remote-sensing inputs.
        #   Real examples: ScaleMAE, DOFA, Satlas Swin, timm ImageNet models.
        super().__init__(bands=bands, normalization="bandspec_zscore")
        #
        # Option B — "identity" (backbone handles normalization internally):
        #   Framework passes raw sensor values unchanged.
        #   Real example: OlmoEarth — its internal Normalizer consumes raw
        #   DN/reflectance; a second z-score would corrupt it.
        #   Pattern: change the line above to normalization="identity"
        #
        # Option C — "model_native" (exact pretrain scale is known):
        #   Framework unit-converts to your backbone's expected input scale,
        #   then applies declared per-channel mean/std.
        #   Real examples: Prithvi-EO (S2_DN), Clay v1.5 / CROMA (REFLECTANCE_0_1)
        #   Pattern: declare class attributes BEFORE the super().__init__ call:
        #
        #     from torchgeo_bench.models._input_units import InputUnit
        #     class MyGeoFM(BenchModel):
        #         expected_input_unit = InputUnit.REFLECTANCE_0_1  # or S2_DN
        #         pretrain_mean = [0.485, 0.456, 0.406]  # optional, per channel
        #         pretrain_std  = [0.229, 0.224, 0.225]  # optional, per channel
        #         def __init__(self, bands, ...):
        #             super().__init__(bands=bands, normalization="model_native")
        # self.num_channels == len(bands) is now available.
        # Each BandSpec also carries .wavelength_um, .sensor, and .name —
        # extract and cache anything your backbone needs at forward time here.
        # See "Accessing band metadata" in docs/user/eval_own_model.rst.

        # TODO: replace the stub below with your backbone construction.
        #
        # Weight loading via HuggingFace Hub (recommended):
        #   from huggingface_hub import hf_hub_download
        #   ckpt = hf_hub_download(repo_id="my-org/my-model", filename="weights.pt")
        #   self.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"))
        #
        self.backbone = nn.Identity()  # TODO: replace with your backbone
        logger.info(
            "MyGeoFM initialized with %d input channels (pretrained=%s)",
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
        # TODO: call your backbone here. and pool to (B, K) if necessary.  Examples:
        #
        # CNN / global-pooled backbone (already returns (B, K)):
        #   return self.backbone(images)
        #
        # ViT with CLS token (output (B, N+1, D) — token 0 is CLS):
        #   tokens = self.backbone(images)
        #   return tokens[:, 0, :]
        #
        # ViT without CLS — average patch tokens:
        #   tokens = self.backbone(images)   # (B, N, D)
        #   return tokens.mean(dim=1)
        #
        x = self.backbone(images)
        if x.ndim == 4:  # (B, K, H, W) — pool spatial dims
            x = x.flatten(start_dim=2).mean(dim=-1)
        return x  # (B, K)
