"""Kuro Siwo (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="kuro_siwo",
    task="segmentation",
    num_classes=4,
    multilabel=False,
    rgb_bands=("vv", "vh"),
    split_sizes=SplitSizes(train=4000, val=1000, test=2000),
    source=V2Source(
        "GeoBenchKuroSiwo",
        validation_split="val",
        band_order_strategy="by_sensor",
        sample_adapter="post_sar_dem",
        canonical_sensor_order=("sar", "dem"),
        return_stacked_image=False,
        time_step=("post",),
    ),

    bands=(
        BandSpec("sar", "vv", "vv", mean=0.1347, std=1.0677, min=0, max=2550.89),
        BandSpec("sar", "vh", "vh", mean=0.0273, std=0.1723, min=0, max=530.453),
        BandSpec("dem", "dem", "dem", mean=146.235, std=465.777, min=-32768, max=1690.83),
    ),
)
