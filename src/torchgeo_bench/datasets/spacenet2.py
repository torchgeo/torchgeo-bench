"""SpaceNet2 (GeoBench V2) benchmark dataset."""

import torch

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class SpaceNet2(_V2Dataset):
    """WorldView building footprint segmentation (2 classes).

    8 multispectral + 1 panchromatic band from WorldView satellite.

    The task is natively 2-class ``{0: background/no-building, 1: building}``.
    Upstream GeoBench applies ``mask = mask + 1`` (*"to have a true background
    class"*), declaring 3 classes ``('background', 'no-building', 'building')``
    and shipping masks valued ``{1, 2}`` — the added ``background`` class 0 never
    actually occurs. That vestigial class 0 gave segmentation heads a
    never-correct label to escape to on low-signal tiles (it dominated the
    label-quality audit's "most suspect" ranking). We reverse the upstream
    offset in :meth:`canonicalize_sample` (``mask - 1``), recovering the native
    2-class labels ``{0: no-building, 1: building}``.

    See https://github.com/The-AI-Alliance/GEO-Bench-2 (geobench_v2/datasets/
    spacenet2.py: ``sample["mask"] = ... + 1``).
    """

    band_order_strategy = "by_sensor"

    name = "spacenet2"
    task = "segmentation"
    num_classes = 2
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 5186, "val": 1461, "test": 2961}

    # fmt: off
    bands = [
        BandSpec("worldview", "coastal", "coastal", mean=296.081, std=107.482, min=0, max=1227),
        BandSpec("worldview", "blue", "blue", mean=357.957, std=151.518, min=0, max=1570),
        BandSpec("worldview", "green", "green", mean=465.239, std=229.433, min=0, max=2047),
        BandSpec("worldview", "yellow", "yellow", mean=417.796, std=230.014, min=0, max=2047),
        BandSpec("worldview", "red", "red", mean=334.455, std=198.499, min=0, max=1933),
        BandSpec("worldview", "red_edge", "red_edge", mean=409.533, std=212.211, min=0, max=2047),
        BandSpec("worldview", "nir1", "nir1", mean=481.216, std=240.981, min=0, max=2047),
        BandSpec("worldview", "nir2", "nir2", mean=364.308, std=196.878, min=0, max=2047),
        BandSpec("pan", "pan", "pan", mean=469.092, std=266.975, min=0, max=2047),
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
