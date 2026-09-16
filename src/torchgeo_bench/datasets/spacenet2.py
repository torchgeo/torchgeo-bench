"""SpaceNet2 (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="spacenet2",
    task="segmentation",
    num_classes=2,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=5186, val=1461, test=2961),
    source=V2Source(
        "GeoBenchSpaceNet2", band_order_strategy="by_sensor", sample_adapter="offset_mask",
        align_to_output=True,
    ),

    bands=(
        BandSpec("worldview", "coastal", "coastal", mean=296.081, std=107.482, min=0, max=1227),
        BandSpec("worldview", "blue", "blue", mean=357.957, std=151.518, min=0, max=1570),
        BandSpec("worldview", "green", "green", mean=465.239, std=229.433, min=0, max=2047),
        BandSpec("worldview", "yellow", "yellow", mean=417.796, std=230.014, min=0, max=2047),
        BandSpec("worldview", "red", "red", mean=334.455, std=198.499, min=0, max=1933),
        BandSpec("worldview", "red_edge", "red_edge", mean=409.533, std=212.211, min=0, max=2047),
        BandSpec("worldview", "nir1", "nir1", mean=481.216, std=240.981, min=0, max=2047),
        BandSpec("worldview", "nir2", "nir2", mean=364.308, std=196.878, min=0, max=2047),
        BandSpec("pan", "pan", "pan", mean=469.092, std=266.975, min=0, max=2047),
    ),
)
