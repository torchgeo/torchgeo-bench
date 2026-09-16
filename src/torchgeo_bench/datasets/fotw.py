"""Fields of the World (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="fotw",
    task="segmentation",
    num_classes=4,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    default_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=4000, val=1000, test=2000),
    source=V2Source("GeoBenchFieldsOfTheWorld", sample_adapter="later_acquisition"),

    bands=(
        # Sentinel-2 centre wavelengths (B04/B03/B02/B08).
        BandSpec("s2", "red", "red", mean=937.509, std=807.662, min=0, max=17499, wavelength_um=0.665),
        BandSpec("s2", "green", "green", mean=923.717, std=677.861, min=0, max=17653, wavelength_um=0.56),
        BandSpec("s2", "blue", "blue", mean=678.358, std=645.035, min=0, max=20214, wavelength_um=0.49),
        BandSpec("s2", "nir", "nir", mean=3028.48, std=1037.38, min=0, max=17200, wavelength_um=0.842),
    ),
)
