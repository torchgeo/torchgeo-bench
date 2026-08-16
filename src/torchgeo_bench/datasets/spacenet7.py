"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _OffsetMaskV2Dataset


class SpaceNet7(_OffsetMaskV2Dataset):
    """Planet building footprint segmentation (2 classes).

    RGB imagery from Planet satellites. Upstream ships masks valued ``{1, 2}``;
    we restore the native ``{0, 1}`` labels — see
    :meth:`~._OffsetMaskV2Dataset.canonicalize_sample`. Despite upstream's
    "multi-temporal" framing, the benchmark ships one image and one mask per
    sample, so this is single-image segmentation rather than change detection.
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
