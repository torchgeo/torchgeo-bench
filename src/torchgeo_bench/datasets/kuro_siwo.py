"""Kuro Siwo (GeoBench V2) benchmark dataset."""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .base import BandSpec
from .geobench_v2 import GeoBenchv2, _V2Dataset


class KuroSiwo(_V2Dataset):
    """SAR flood mapping segmentation (4 classes).

    Upstream emits multi-temporal SAR (``image_pre_1`` / ``image_pre_2`` /
    ``image_post``) plus a static DEM (``image_dem``).  Its built-in
    ``return_stacked_image=True`` path stacks per-timestep tensors along a
    new temporal axis, which (a) leaves the result 4-D ``(C, T, H, W)`` and
    (b) hits an assertion when SAR and DEM channel counts differ.

    To produce a clean 3-D ``(C, H, W)`` image we bypass that path
    altogether: we ask upstream for the post-event SAR only
    (``time_step=["post"]``), then concatenate optional DEM along the
    channel dimension ourselves in :meth:`canonicalize_sample`.
    """

    band_order_strategy = "by_sensor"

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

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return a :class:`GeoBenchv2` configured for single-timestep SAR + DEM.

        Forces ``return_stacked_image=False`` (so the upstream emits per-modality
        keys we can stack ourselves) and ``time_step=["post"]`` (so only the
        post-event SAR acquisition is loaded). :meth:`canonicalize_sample`
        then folds the per-modality tensors into a single 3-D ``image`` key.
        """
        del partition
        band_order = self.build_band_order(bands)
        canonicalize = self.canonicalize_sample

        def chained(sample: dict) -> dict:
            sample = canonicalize(sample)
            if transform is not None:
                sample = transform(sample)
            return sample

        return GeoBenchv2(
            root=self.data_root(),
            dataset_name=self.name,
            split=split,
            band_order=band_order,
            transforms=chained,
            data_normalizer=nn.Identity,
            time_step=["post"],
        )

    def canonicalize_sample(self, sample: dict) -> dict:
        """Fold per-modality keys into a single 3-D ``(C, H, W)`` image tensor.

        Upstream emits ``image_post`` for SAR (we only request the post-event
        timestep) and/or ``image_dem`` depending on the requested band order.
        Both are 3-D ``(C, H, W)`` so we can simply concatenate them along
        the channel dimension. Per-modality keys are removed from the sample
        once merged.
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
