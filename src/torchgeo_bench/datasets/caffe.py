"""CaFFe (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="caffe",
    task="segmentation",
    num_classes=4,
    multilabel=False,
    default_bands=("gray",),
    rgb_bands=None,
    split_sizes=SplitSizes(train=4000, val=1000, test=2000),
    source=V2Source("GeoBenchCaFFe"),

    bands=(
        BandSpec("aerial", "gray", "gray", mean=68.4868, std=82.7774, min=0, max=255),
    ),
)
