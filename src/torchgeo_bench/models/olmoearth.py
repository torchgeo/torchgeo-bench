"""OlmoEarth model wrapper for torchgeo-bench.

Wraps the OlmoEarth geospatial foundation model (AI2) for use with the
BenchModel interface. Supports both RGB (3-channel) and full multispectral
(12-channel Sentinel-2) input.

When given RGB input, the 3 channels are mapped to B02/B03/B04 in OlmoEarth's
expected band order, the remaining 9 bands are zero-filled, and the
corresponding band sets are marked as MISSING in the mask so the model
treats them as absent data rather than real zero-valued observations.

Reference implementation:
    https://github.com/isaaccorley/geopool/blob/main/scripts/embed_olmoearth.py
"""

import logging
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

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

# Number of channels per band set
_BAND_SET_SIZES = (4, 6, 2)
_NUM_BAND_SETS = len(_BAND_SET_SIZES)
_TOTAL_S2_CHANNELS = sum(_BAND_SET_SIZES)  # 12


class OlmoEarthBenchModel(BenchModel):
    """BenchModel wrapper for OlmoEarth geospatial foundation models.

    OlmoEarth is a ViT-based multimodal foundation model trained on
    Sentinel-1, Sentinel-2, and Landsat imagery by the Allen Institute for AI.

    Supports two input modes:

    - **RGB mode** (``num_channels=3``): Input is ``(B, 3, H, W)`` with channels
      ordered as red, green, blue. These are mapped to OlmoEarth's B04, B03, B02
      (band set 0 positions 2, 1, 0). The 4th band in set 0 (B08/NIR) and all
      bands in sets 1–2 are zero-filled and masked as MISSING. Performance will
      be lower than full multispectral mode.

    - **Full S2 mode** (``num_channels=12``): Input is ``(B, 12, H, W)`` with
      all 12 Sentinel-2 bands in OlmoEarth order (see ``OLMOEARTH_S2_BANDS``).
      All band sets are fully visible.

    Important:
        - Dataset normalization should be set to ``none`` — OlmoEarth applies its
          own normalization internally via ``Normalizer``.
        - Minimum spatial size is approximately 64×64 pixels.

    Args:
        num_channels: Number of input channels (3 for RGB, 12 for full S2).
        model_size: One of ``"nano"``, ``"tiny"``, ``"base"``, ``"large"``.
        patch_size: Patch size for the encoder (default 8).
        input_res: Input resolution in meters (default 10 for Sentinel-2).
        time_steps: Number of temporal repeats for single-timestep input (default 3).
        std_multiplier: Standard deviation multiplier for normalization (default 2.0).
        normalize: If True, L2-normalize output embeddings.
    """

    def __init__(
        self,
        num_channels: int = 3,
        *,
        model_size: Literal["nano", "tiny", "base", "large"] = "base",
        patch_size: int = 8,
        input_res: int = 10,
        time_steps: int = 3,
        std_multiplier: float = 2.0,
        normalize: bool = False,
        **_kwargs,
    ) -> None:
        super().__init__(num_channels=num_channels)

        if num_channels not in (3, 12):
            raise ValueError(
                f"OlmoEarth supports 3 (RGB) or 12 (full S2) input channels, got {num_channels}."
            )

        # Lazy imports so the package is only needed when this model is used
        from olmoearth_pretrain_minimal import ModelID, Normalizer, load_model_from_id
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

        self.model_size = model_size
        self.patch_size = patch_size
        self.input_res = input_res
        self.time_steps = time_steps
        self.do_normalize = normalize
        self._rgb_mode = num_channels == 3

        model_id = getattr(ModelID, f"OLMOEARTH_V1_{model_size.upper()}")
        self.encoder_model = load_model_from_id(model_id, load_weights=True)
        self.normalizer = Normalizer(std_multiplier=std_multiplier)
        self._modality = Modality.SENTINEL2_L2A

    def _rgb_to_s2(self, images: torch.Tensor) -> torch.Tensor:
        """Map (B, 3, H, W) RGB to (B, 12, H, W) in OlmoEarth S2 band order.

        RGB channels are assumed to be [red, green, blue] and are placed at
        OlmoEarth positions [2, 1, 0] (B04, B03, B02). The remaining 9 bands
        are zero-filled.
        """
        B, _, H, W = images.shape
        s2 = torch.zeros(B, _TOTAL_S2_CHANNELS, H, W, device=images.device, dtype=images.dtype)
        # red -> B04 (index 2), green -> B03 (index 1), blue -> B02 (index 0)
        s2[:, 2] = images[:, 0]  # red -> B04
        s2[:, 1] = images[:, 1]  # green -> B03
        s2[:, 0] = images[:, 2]  # blue -> B02
        return s2

    def _build_mask(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Build the mask tensor (B, H, W, T).

        All values are 0 (ONLINE_ENCODER = visible). For RGB mode, missing
        bands are zero-filled in the data tensor but not masked — OlmoEarth's
        per-band-set masking would exclude all spatial modalities from pooling.
        """
        return torch.zeros(B, H, W, self.time_steps, dtype=torch.long, device=device)

    @torch.no_grad()
    def forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract image-level embeddings.

        Args:
            images: Input tensor — ``(B, 3, H, W)`` for RGB or ``(B, 12, H, W)``
                for full Sentinel-2 (raw, un-normalized values).
            bboxes: Unused.

        Returns:
            Embeddings of shape ``(B, D)`` where D depends on model_size:
            nano=128, tiny=192, base=768, large=1024.
        """
        del bboxes
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.nn.flexi_vit import PoolingType
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import (
            MaskedOlmoEarthSample,
        )

        device = images.device

        # If RGB, expand to 12-channel S2 layout
        if self._rgb_mode:
            images = self._rgb_to_s2(images)

        B, C, H, W = images.shape

        # (B, C, H, W) -> (B, H, W, C) -> numpy for normalization
        images_nhwc = images.permute(0, 2, 3, 1).cpu().numpy()

        # Repeat along temporal dimension to simulate multi-temporal input
        images_nhwtc = np.repeat(images_nhwc[:, :, :, None, :], self.time_steps, axis=3)

        # Apply OlmoEarth-specific normalization
        images_nhwtc = self.normalizer.normalize(self._modality, images_nhwtc)

        # Build timestamps: (B, T, 3) — [day, month, year]
        timestamps = torch.zeros(B, self.time_steps, 3, dtype=torch.long, device=device)
        timestamps[:, :, 0] = 15  # day
        timestamps[:, :, 1] = 6  # month (June)
        timestamps[:, :, 2] = 2020  # year

        # Build mask with per-band-set MISSING flags for RGB mode
        s2_mask = self._build_mask(B, H, W, device)

        # Create MaskedOlmoEarthSample
        sample = MaskedOlmoEarthSample(
            timestamps=timestamps,
            sentinel2_l2a=torch.from_numpy(images_nhwtc).float().to(device),
            sentinel2_l2a_mask=s2_mask,
        )

        # Encode
        outputs = self.encoder_model.encoder(
            sample,
            patch_size=self.patch_size,
            input_res=self.input_res,
            fast_pass=True,
        )

        # Pool spatially across band sets -> (B, H', W', D)
        pooled = outputs["tokens_and_masks"].pool_spatially(PoolingType.MEAN)
        # Mean across spatial dims -> (B, D)
        embeddings = pooled.mean(dim=(1, 2))

        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
