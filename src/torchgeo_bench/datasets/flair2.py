"""FLAIR2 (GeoBench V2) benchmark dataset."""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, SplitSizes, V2Source

# fmt: off
SPEC = DatasetSpec(
    name="flair2",
    task="segmentation",
    num_classes=13,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=4049, val=1022, test=3022),
    source=V2Source("GeoBenchFLAIR2"),

    bands=(
        # IGN BD ORTHO centre wavelengths (R/G/B/NIR); elevation is non-spectral.
        BandSpec("aerial", "red", "red", mean=111.395, std=51.2846, min=0, max=255, wavelength_um=0.66),
        BandSpec("aerial", "green", "green", mean=115.788, std=45.1, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "blue", mean=106.896, std=44.4006, min=0, max=255, wavelength_um=0.48),
        BandSpec("aerial", "nir", "nir", mean=104.085, std=39.566, min=0, max=255, wavelength_um=0.83),
        BandSpec("elevation", "elevation", "elevation", mean=17.7749, std=30.34, min=0, max=255),
    ),
)
