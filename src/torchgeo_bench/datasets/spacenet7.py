"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class SpaceNet7(_V2Dataset):
    """Planet building change segmentation (3 classes).

    RGB imagery from Planet satellites.
    """

    name = "spacenet7"
    task = "segmentation"
    num_classes = 3
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
