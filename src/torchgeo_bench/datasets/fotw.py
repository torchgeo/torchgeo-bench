"""Fields of the World (GeoBench V2) benchmark dataset."""

from .base import BandSpec
from .geobench_v2 import _V2Dataset


class FieldsOfTheWorld(_V2Dataset):
    """Sentinel-2 field boundary segmentation (4 classes).

    Classes: background, field, boundary, other. Upstream returns
    ``image_a`` / ``image_b`` change-detection pairs;
    :meth:`canonicalize_sample` keeps the later acquisition (``image_b``).
    """

    name = "fotw"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 4000, "val": 1000, "test": 2000}

    # fmt: off
    bands = [
        BandSpec("s2", "red", "red", mean=937.509, std=807.662, min=0, max=17499),
        BandSpec("s2", "green", "green", mean=923.717, std=677.861, min=0, max=17653),
        BandSpec("s2", "blue", "blue", mean=678.358, std=645.035, min=0, max=20214),
        BandSpec("s2", "nir", "nir", mean=3028.48, std=1037.38, min=0, max=17200),
    ]
    # fmt: on

    def canonicalize_sample(self, sample: dict) -> dict:
        """Pick the later acquisition (``image_b``) and surface it as ``image``."""
        if "image" not in sample and "image_b" in sample:
            sample["image"] = sample.pop("image_b")
            sample.pop("image_a", None)
        return sample
