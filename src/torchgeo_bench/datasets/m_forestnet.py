"""MForestnet (GeoBench V1) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetCapabilities, DatasetSpec, SplitSizes, V1Source

# fmt: off
SPEC = DatasetSpec(
    name="m-forestnet",
    task="classification",
    num_classes=12,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    default_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=6464, val=989, test=993),
    source=V1Source(),
    capabilities=DatasetCapabilities(supports_partitions=True),

    bands=(
        BandSpec("landsat", "blue", "02 - Blue", mean=72.376, std=16.2839, min=0, max=255, wavelength_um=0.49),
        BandSpec("landsat", "green", "03 - Green", mean=83.1816, std=15.3587, min=0, max=255, wavelength_um=0.56),
        BandSpec("landsat", "red", "04 - Red", mean=77.0862, std=16.6665, min=0, max=255, wavelength_um=0.665),
        BandSpec("landsat", "nir", "05 - NIR", mean=123.543, std=16.9485, min=0, max=250),
        BandSpec("landsat", "swir_1", "06 - SWIR1", mean=91.0484, std=14.2801, min=0, max=255),
        BandSpec("landsat", "swir_2", "07 - SWIR2", mean=74.3097, std=13.2854, min=0, max=255),
    ),
)
