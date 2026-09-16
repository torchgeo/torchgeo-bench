"""Burn Scars (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="burn_scars",
    task="segmentation",
    num_classes=3,
    multilabel=False,
    rgb_bands=("b04", "b03", "b02"),
    default_bands=("b04", "b03", "b02"),
    split_sizes=SplitSizes(train=524, val=160, test=120),
    source=V2Source("GeoBenchBurnScars"),

    bands=(
        BandSpec("s2", "b02", "B02", mean=0.0526, std=0.0308, min=0, max=1, wavelength_um=0.49),
        BandSpec("s2", "b03", "B03", mean=0.078, std=0.0376, min=0, max=1, wavelength_um=0.56),
        BandSpec("s2", "b04", "B04", mean=0.0947, std=0.0549, min=0, max=1, wavelength_um=0.665),
        BandSpec("s2", "b8a", "B8A", mean=0.2139, std=0.0701, min=0, max=1, wavelength_um=0.865),
        BandSpec("s2", "b11", "B11", mean=0.2356, std=0.0911, min=0, max=1, wavelength_um=1.61),
        BandSpec("s2", "b12", "B12", mean=0.171, std=0.0836, min=0, max=1, wavelength_um=2.19),
    ),
)
