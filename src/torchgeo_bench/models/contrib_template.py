"""Copy-paste skeleton for external model contributions.

This file is **not** part of the public ``torchgeo_bench.models`` namespace.
External contributors should copy this file to their own working directory,
fill in the TODO sections, and follow the contribution guide at
``docs/user/eval_own_model.rst`` (Stage 1) or
``docs/user/contribute_model.rst`` (Stage 2).

Two classes are provided:

* :class:`MyGeoFM` — standard case: use the default ``bandspec_zscore``
  normalization (recommended for most remote-sensing backbones).
* :class:`MyGeoFMInternal` — identity-normalization variant: override
  ``normalize_inputs`` to return the tensor unchanged when the backbone
  handles all preprocessing internally (e.g. it ships its own ``Normalizer``
  module or always expects raw DN values).
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

    Normalization strategy: ``bandspec_zscore`` (default) applies per-channel
    ``(x - mean) / std`` using the dataset's :class:`BandSpec` statistics.
    This is the correct choice for most remote-sensing backbones that were
    trained on normalized inputs — it produces ~N(0, 1) features regardless
    of the source sensor unit.

    For the ``model_native`` strategy (faithful to the backbone's training
    pipeline), declare ``expected_input_unit``, ``pretrain_mean``, and
    ``pretrain_std`` as class attributes *before* calling ``super().__init__``.
    For the ``identity`` strategy (backbone handles normalization), use
    :class:`MyGeoFMInternal` instead.

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
        # TODO: add any extra kwargs your backbone needs; mirror them in your
        #       Hydra YAML under src/torchgeo_bench/conf/model/<name>.yaml
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands)
        # self.num_channels == len(bands) is now available.

        # TODO: replace the stub below with your backbone construction.
        #
        # Weight loading patterns:
        #   from huggingface_hub import hf_hub_download
        #   ckpt = hf_hub_download(repo_id="my-org/my-model", filename="weights.pt")
        #   self.backbone.load_state_dict(torch.load(ckpt, map_location="cpu"))
        #
        # For ViT-style backbones, decide whether to use the CLS token or
        # spatially average the patch tokens — see _forward_patch_features below.
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
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return embeddings ``(B, K)`` from already-normalized inputs.

        ``images`` has shape ``(B, C, H, W)`` and has already been passed
        through ``self.normalize_inputs`` by the sealed ``forward_patch_features``.

        Args:
            images: Normalized input tensor ``(B, C, H, W)``.
            bboxes: Optional bounding boxes ``(B, 4)``; most backbones ignore
                this and should ``del bboxes``.

        Returns:
            Embedding tensor of shape ``(B, K)``.
        """
        del bboxes
        # TODO: call your backbone here.
        #
        # CNN / pooled backbone (returns (B, K) directly):
        #   return self.backbone(images)
        #
        # ViT with CLS token (output shape (B, N+1, D)):
        #   tokens = self.backbone(images)   # (B, N+1, D)
        #   return tokens[:, 0, :]           # CLS token → (B, D)
        #
        # ViT without CLS — average patch tokens:
        #   tokens = self.backbone(images)   # (B, N, D)
        #   return tokens.mean(dim=1)        # (B, D)
        x = self.backbone(images)
        # Flatten spatial dims if backbone returns (B, K, H, W).
        if x.ndim == 4:
            x = x.flatten(start_dim=2).mean(dim=-1)
        return x  # (B, K)


class MyGeoFMInternal(BenchModel):
    """Template variant for backbones that handle normalization internally.

    Use this class when your backbone ships its own ``Normalizer`` layer,
    always expects raw sensor values, or applies its own per-channel statistics
    inside the forward pass.  Overriding ``normalize_inputs`` to the identity
    ensures the framework does **not** apply its own z-score on top.

    In all other respects, this class is identical to :class:`MyGeoFM`.

    Args:
        bands: Ordered list of :class:`BandSpec` from the dataset wrapper.
        pretrained: Load pretrained weights (default: ``True``).
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        pretrained: bool = True,
        **_kwargs: object,
    ) -> None:
        # Pass normalization="identity" so BenchModel builds a no-op normalizer.
        # This means forward_patch_features passes raw inputs to _forward_patch_features.
        super().__init__(bands=bands, normalization="identity")
        self.backbone = nn.Identity()  # TODO: replace with your backbone
        logger.info(
            "MyGeoFMInternal initialized with %d input channels (pretrained=%s)",
            self.num_channels,
            pretrained,
        )

    # normalize_inputs is inherited from BenchModel and will return the tensor
    # unchanged because normalization="identity" was passed to super().__init__.
    # No need to override it explicitly.

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return embeddings ``(B, K)`` from raw (un-normalized) inputs.

        ``images`` is the raw sensor tensor; ``normalize_inputs`` is a no-op
        for this class.

        Args:
            images: Raw input tensor ``(B, C, H, W)`` (not normalized).
            bboxes: Optional bounding boxes ``(B, 4)``; most backbones ignore
                this and should ``del bboxes``.

        Returns:
            Embedding tensor of shape ``(B, K)``.
        """
        del bboxes
        # TODO: call your backbone here (same patterns as MyGeoFM).
        x = self.backbone(images)
        if x.ndim == 4:
            x = x.flatten(start_dim=2).mean(dim=-1)
        return x  # (B, K)


# Expose a helper for contributors to quickly verify their subclass works.
def _check_template(cls: type[BenchModel], num_channels: int = 3) -> None:
    """Smoke-test a BenchModel subclass with random data.

    Useful during development; not imported by the package.

    Args:
        cls: A :class:`BenchModel` subclass to test.
        num_channels: Number of input channels.
    """
    bands = [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=500.0,
            std=100.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(num_channels)
    ]
    model = cls(bands=bands)
    x = torch.randn(2, num_channels, 64, 64)
    with torch.no_grad():
        out = model.forward_patch_features(x)
    assert out.ndim == 2 and out.shape[0] == 2, f"Expected (2, K), got {out.shape}"
    logger.info("%s smoke-test passed — output shape %s", cls.__name__, tuple(out.shape))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _check_template(MyGeoFM)
    _check_template(MyGeoFMInternal)
    logger.info("All template smoke tests passed.")
