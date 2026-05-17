"""OlmoEarth model wrapper for torchgeo-bench.

Wraps the OlmoEarth geospatial foundation model (AI2) for use with the
BenchModel interface. Supports both RGB (3-channel) and full multispectral
(12-channel Sentinel-2) input.

When given RGB input, the 3 channels are mapped to B02/B03/B04 in OlmoEarth's
expected band order; the remaining 9 bands are zero-filled.  All band sets
remain marked visible in the mask — the encoder learns to attend through
zero-valued bands but the docstring's earlier claim of MISSING masking was
never actually implemented (and breaks ``pool_spatially`` when applied).

Reference implementations (canonical first):
    https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Inference-Quickstart.md
    https://github.com/isaaccorley/geopool/blob/main/scripts/embed_olmoearth.py
"""

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

logger = logging.getLogger(__name__)

# Sentinel-2 band order expected by OlmoEarth
# Band set 0: B02, B03, B04, B08  (4 bands — includes RGB)
# Band set 1: B05, B06, B07, B8A, B11, B12  (6 bands)
# Band set 2: B01, B09  (2 bands)
OLMOEARTH_S2_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
)

# Number of channels per band set (B02/B03/B04/B08 — B05-B12 — B01/B09).
_BAND_SET_SIZES = (4, 6, 2)
_NUM_BAND_SETS = len(_BAND_SET_SIZES)  # 3
_TOTAL_S2_CHANNELS = sum(_BAND_SET_SIZES)  # 12


class OlmoEarthBenchModel(BenchModel):
    """BenchModel wrapper for OlmoEarth geospatial foundation models.

    OlmoEarth is a ViT-based multimodal foundation model trained on
    Sentinel-1, Sentinel-2, and Landsat imagery by the Allen Institute for AI.

    Supports two input modes (chosen by ``len(bands)``):

    - **RGB mode** (``len(bands) == 3``): input is mapped to OlmoEarth's
      B04, B03, B02 (band set 0 positions 2, 1, 0).  The 4th band in set 0
      (B08/NIR) and all bands in sets 1–2 are zero-filled.  Performance will
      be lower than full multispectral mode.
    - **Full S2 mode** (``len(bands) == 12``): all 12 Sentinel-2 bands are
      forwarded in OlmoEarth order (see :data:`OLMOEARTH_S2_BANDS`).

    The wrapper overrides :meth:`normalize_inputs` to identity — OlmoEarth's
    internal :class:`Normalizer` consumes raw values directly.

    Args:
        bands: Either a 3-band RGB list or a 12-band Sentinel-2 list.
        model_size: One of ``"nano"``, ``"tiny"``, ``"base"``, ``"large"``.
        patch_size: Patch size for the encoder (default 4 per the AI2 quickstart;
            smaller patch sizes generally perform better but use more GPU memory).
        input_res: Input resolution in meters (default 10 for Sentinel-2).
        time_steps: Number of temporal slots in the input.  Default 1, matching the
            official ``olmoearth_pretrain`` quickstart for single-image inference.
            Replicating a single image across T>1 timesteps was historically used
            here to satisfy a buggy 4-D mask shape, but the correct fix is a 5-D
            mask and T=1 (see ``_build_mask`` below).
        std_multiplier: Standard deviation multiplier for normalization.
        normalize: If True, L2-normalize output embeddings.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        model_size: Literal["nano", "tiny", "base", "large"] = "base",
        patch_size: int = 4,
        input_res: int = 10,
        time_steps: int = 1,
        std_multiplier: float = 2.0,
        normalize: bool = False,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)

        if self.num_channels not in (3, 12):
            raise ValueError(
                "OlmoEarth supports 3 (RGB) or 12 (full S2) input channels, "
                f"got {self.num_channels}."
            )

        # Lazy imports so the package is only needed when this model is used
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

        # The package logs full ModalitySpec dumps at INFO level on every
        # encoder construction — useful when developing OlmoEarth itself,
        # noisy for every torchgeo-bench task.  Demote to WARNING.
        for logger_name in (
            "olmoearth_pretrain_minimal",
            "olmoearth_pretrain_minimal.olmoearth_pretrain_v1",
        ):
            logging.getLogger(logger_name).setLevel(logging.WARNING)

        self.model_size = model_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.time_steps = time_steps
        self.do_normalize = normalize
        self._rgb_mode = self.num_channels == 3

        model_id = getattr(ModelID, f"OLMOEARTH_V1_{model_size.upper()}")
        self.encoder_model = load_model_from_id(model_id, load_weights=True)
        self.normalizer = Normalizer(std_multiplier=std_multiplier)
        self._modality = Modality.SENTINEL2_L2A

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Identity — OlmoEarth's internal :class:`Normalizer` handles raw values."""
        return images

    def _rgb_to_s2(self, images: torch.Tensor) -> torch.Tensor:
        """Map (B, 3, H, W) RGB to (B, 12, H, W) in OlmoEarth S2 band order."""
        B, _, H, W = images.shape
        s2 = torch.zeros(B, _TOTAL_S2_CHANNELS, H, W, device=images.device, dtype=images.dtype)
        # red -> B04 (index 2), green -> B03 (index 1), blue -> B02 (index 0)
        s2[:, 2] = images[:, 0]
        s2[:, 1] = images[:, 1]
        s2[:, 0] = images[:, 2]
        return s2

    def _build_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Build the per-band-set mask tensor — all visible.

        Shape is ``(B, H, W, T, num_band_sets)`` per AI2's Inference-Quickstart:
        the encoder's ``apply_embedding_to_modality`` slices ``mask[..., idx]``
        for each band-set index, so the last dimension MUST be ``num_band_sets``
        (3 for Sentinel-2: (B02/B03/B04/B08), (B05-B12 minus B10), (B01/B09)).

        The previous 4-D ``(B, H, W, T)`` mask only worked because ``T`` happened
        to equal ``num_band_sets=3``; with the canonical ``T=1`` it would index
        out of bounds.  ``ONLINE_ENCODER`` has value 0, so all-zeros == all
        visible.
        """
        return torch.zeros(
            B, H, W, self.time_steps, _NUM_BAND_SETS, dtype=torch.float32, device=device
        )

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract image-level embeddings from raw inputs."""
        del bboxes
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        device = images.device

        if self._rgb_mode:
            images = self._rgb_to_s2(images)

        B, C, H, W = images.shape

        # Layout: (B, H, W, C) -> (B, H, W, T, C).  np.repeat only when T>1 so
        # the common T=1 path doesn't allocate a wasted copy.
        images_nhwc = images.permute(0, 2, 3, 1).cpu().numpy()
        images_nhwtc = images_nhwc[:, :, :, None, :]
        if self.time_steps > 1:
            images_nhwtc = np.repeat(images_nhwtc, self.time_steps, axis=3)
        images_nhwtc = self.normalizer.normalize(self._modality, images_nhwtc)

        timestamps = torch.zeros(B, self.time_steps, 3, dtype=torch.long, device=device)
        timestamps[:, :, 0] = 15
        timestamps[:, :, 1] = 6
        timestamps[:, :, 2] = 2020

        s2_mask = self._build_mask(B, H, W, device)

        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=torch.from_numpy(images_nhwtc).float().to(device),
            sentinel2_l2a_mask=s2_mask,
        )

        outputs = self.encoder_model.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=True,
        )

        pooled = outputs["tokens_and_masks"].pool_spatially(PoolingType.MEAN)
        embeddings = pooled.mean(dim=(1, 2))

        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
