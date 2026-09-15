"""SpaceNet7 (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="spacenet7",
    task="segmentation",
    num_classes=2,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=3500, val=652, test=1152),
    source=V2Source("GeoBenchSpaceNet7", sample_adapter="offset_mask"),

    bands=(
        BandSpec("planet", "red", "red", mean=117.85, std=61.9829, min=0, max=255),
        BandSpec("planet", "green", "green", mean=104.531, std=49.7879, min=0, max=255),
        BandSpec("planet", "blue", "blue", mean=77.561, std=46.01, min=0, max=255),
    ),
)
