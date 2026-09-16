"""MPv4ger (GeoBench V1) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetCapabilities, DatasetSpec, SplitSizes, V1Source

# fmt: off
SPEC = DatasetSpec(
    name="m-pv4ger",
    task="classification",
    num_classes=2,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    default_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=11814, val=999, test=999),
    source=V1Source(),
    capabilities=DatasetCapabilities(supports_partitions=True),

    bands=(
        # NAIP centre wavelengths (4-band sensor; this dataset only ships RGB).
        BandSpec("aerial", "blue", "Blue", mean=116.316, std=44.5176, min=2, max=254, wavelength_um=0.45),
        BandSpec("aerial", "green", "Green", mean=119.375, std=48.1189, min=7, max=254, wavelength_um=0.55),
        BandSpec("aerial", "red", "Red", mean=113.102, std=54.0881, min=1, max=254, wavelength_um=0.65),
    ),
)
