"""SpaceNet7 (GeoBench V2) benchmark dataset."""

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class SpaceNet7(_V2Dataset):
    """Planet building footprint segmentation (2 classes).

    RGB imagery from Planet satellites.

    The task is natively 2-class ``{0: no-building, 1: building}``. Upstream
    GeoBench applies ``mask = mask + 1`` (*"to have a true background class"*),
    declaring 3 classes ``('background', 'no-building', 'building')`` and
    shipping masks valued ``{1, 2}`` — the added ``background`` class 0 never
    actually occurs. This is the same defect corrected in
    :class:`~torchgeo_bench.datasets.spacenet2.SpaceNet2`: the vestigial class 0
    gives segmentation heads a never-correct label to escape to on low-signal
    tiles. We reverse the upstream offset in :meth:`canonicalize_sample`
    (``mask - 1``), recovering the native 2-class labels.

    Verified against the shipped tortilla: every mask in all three splits
    (3500 train / 652 val / 1152 test) contains only raw values ``{0, 1}``.

    Despite upstream's "multi-temporal" framing, the benchmark ships one image
    and one mask per sample, so this is single-image segmentation rather than
    change detection.

    See https://github.com/The-AI-Alliance/GEO-Bench-2 (geobench_v2/datasets/
    spacenet7.py: ``mask = torch.from_numpy(mask).long().squeeze(0) + 1``).
    """

    name = "spacenet7"
    task = "segmentation"
    num_classes = 2
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 3500, "val": 652, "test": 1152}

    # fmt: off
    bands = [
        BandSpec("planet", "red", "red", mean=117.85, std=61.9829, min=0, max=255),
        BandSpec("planet", "green", "green", mean=104.531, std=49.7879, min=0, max=255),
        BandSpec("planet", "blue", "blue", mean=77.561, std=46.01, min=0, max=255),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Reverse GeoBench's ``+1`` offset: mask ``1 -> 0``, ``2 -> 1``.

        Upstream ships masks valued ``{1: no-building, 2: building}`` after
        adding 1 to the native ``{0, 1}`` labels. Subtracting one recovers the
        native 2-class labels and drops the never-used ``background`` (0). The
        ``clamp(min=0)`` is defensive: if GeoBench ever emitted its reserved
        class 0, it folds into ``no-building`` (the correct default for a pixel
        with no building) rather than wrapping to ``-1``. Runs before any resize
        transform; masks are integer class ids so the shift is exact.
        """
        mask = sample.get("mask")
        if mask is not None:
            mask = torch.as_tensor(mask)
            sample["mask"] = (mask - 1).clamp_(min=0)
        return sample
