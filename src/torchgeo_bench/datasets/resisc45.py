"""NWPU-RESISC45 aerial scene classification via torchgeo.

31,500 RGB images at 256x256, with 700 images per class, sourced from Google Earth.
8-bit imagery has no per-image geolocation, scale, or radiometric calibration.
Wavelengths are nominal visible-light centres, not measured sensor responses.
Images span 0.2--30 m/px; the ``aerial`` tag's 1 m GSD is only an approximation.
"""

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, GeographySpec, SplitSizes, TorchGeoSource

# fmt: off
SPEC = DatasetSpec(
    name="resisc45",
    task="classification",
    num_classes=45,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=18900, val=6300, test=6300),
    source=TorchGeoSource("RESISC45", root="data/resisc45", download_checksum=True),
    geography=GeographySpec(
        reason=(
            "scene-classification JPEGs with no georeferencing: no sampled image "
            "carries any EXIF block, and the upstream release ships no coordinate table"
        )
    ),

    # Train-split statistics in raw 0-255 units; see scripts/compute_band_statistics.py.
    # ``source_name`` identifies the RGB channel, not a band key in the JPEG.
    bands=(
        BandSpec("aerial", "red", "R", mean=93.8939, std=51.8492, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=97.1123, std=47.2366, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=87.5678, std=47.0631, min=0, max=255, wavelength_um=0.45),
    ),
)
