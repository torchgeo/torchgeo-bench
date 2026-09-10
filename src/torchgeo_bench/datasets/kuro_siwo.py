"""Kuro Siwo (GeoBench V2) benchmark dataset."""

from typing import ClassVar

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class KuroSiwo(_V2Dataset):
    """SAR flood mapping segmentation (4 classes).

    Upstream emits multi-temporal SAR (``image_pre_1`` / ``image_pre_2`` /
    ``image_post``) plus a static DEM (``image_dem``).  Its built-in
    ``return_stacked_image=True`` path stacks per-timestep tensors along a
    new temporal axis, which (a) leaves the result 4-D ``(C, T, H, W)`` and
    (b) hits an assertion when SAR and DEM channel counts differ.

    To produce a clean 3-D ``(C, H, W)`` image we bypass that path
    altogether: :attr:`upstream_kwargs` asks upstream for per-modality keys
    (``return_stacked_image=False``) and the post-event SAR only
    (``time_step=["post"]``), then :meth:`canonicalize_sample` concatenates
    SAR and optional DEM along the channel dimension ourselves.
    """

    band_order_strategy = "by_sensor"
    upstream_kwargs: ClassVar[dict[str, object]] = {
        "return_stacked_image": False,
        "time_step": ["post"],
    }

    name = "kuro_siwo"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands: ClassVar[list[str]] = ["vv", "vh"]
    split_sizes: ClassVar[dict[str, int]] = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands: ClassVar[list[BandSpec]] = [
        BandSpec("sar", "vv", "vv", mean=0.1347, std=1.0677, min=0, max=2550.89),
        BandSpec("sar", "vh", "vh", mean=0.0273, std=0.1723, min=0, max=530.453),
        BandSpec("dem", "dem", "dem", mean=146.235, std=465.777, min=-32768, max=1690.83),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Fold per-modality keys into a single 3-D ``(C, H, W)`` image tensor.

        Upstream emits ``image_post`` for SAR (we only request the post-event
        timestep) and/or ``image_dem`` depending on the requested band order.
        Both are 3-D ``(C, H, W)`` so we can simply concatenate them along
        the channel dimension. Per-modality keys are removed from the sample
        once merged.
        """
        modalities: list[torch.Tensor] = [
            sample.pop(key) for key in ("image_post", "image_dem") if key in sample
        ]
        if modalities:
            sample["image"] = (
                modalities[0] if len(modalities) == 1 else torch.cat(modalities, dim=0)
            )
        return sample
