"""Kuro Siwo (GeoBench V2) benchmark dataset."""

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class KuroSiwo(_V2Dataset):
    """SAR flood mapping segmentation (4 classes).

    Upstream provides SAR at three dates (``image_pre_1`` / ``image_pre_2`` / ``image_post``) plus a static DEM (``image_dem``).

    Its ``return_stacked_image=True`` path produces ``(C, T, H, W)`` tensors and fails when SAR and DEM channel counts differ. Request ``return_stacked_image=False`` and ``time_step=["post"]`` instead, then join post-event SAR and optional DEM into a ``(C, H, W)`` image.
    """

    band_order_strategy = "by_sensor"
    upstream_kwargs = {"return_stacked_image": False, "time_step": ["post"]}

    name = "kuro_siwo"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["vv", "vh"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands = [
        BandSpec("sar", "vv", "vv", mean=0.1347, std=1.0677, min=0, max=2550.89),
        BandSpec("sar", "vh", "vh", mean=0.0273, std=0.1723, min=0, max=530.453),
        BandSpec("dem", "dem", "dem", mean=146.235, std=465.777, min=-32768, max=1690.83),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Join post-event SAR and optional DEM into a ``(C, H, W)`` image.

        Channels from ``image_post`` precede those from ``image_dem``. The original keys are removed after merging.
        """
        modalities: list[torch.Tensor] = []
        for key in ("image_post", "image_dem"):
            if key in sample:
                modalities.append(sample.pop(key))
        if modalities:
            sample["image"] = (
                modalities[0] if len(modalities) == 1 else torch.cat(modalities, dim=0)
            )
        return sample
