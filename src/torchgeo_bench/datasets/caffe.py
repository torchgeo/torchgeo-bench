"""CaFFe (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class CaFFe(_V2Dataset):
    """Aerial grayscale calving-front segmentation (4 classes).

    The upstream GeoBench V2 dataset returns ``(image, mask)`` pairs, so this
    wrapper exposes it as a segmentation task even though the dataset name
    historically suggested classification.
    """

    name = "caffe"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["gray", "gray", "gray"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands = [
        BandSpec("aerial", "gray", "gray", mean=68.4868, std=82.7774, min=0, max=255),
    ]
    # fmt: on
