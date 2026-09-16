"""EuroSAT and spatially disjoint EuroSAT splits from torchgeo."""

from dataclasses import replace

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, GeographySpec, SplitSizes, TorchGeoSource

# fmt: off
SPEC = DatasetSpec(
    name="eurosat",
    task="classification",
    num_classes=10,
    multilabel=False,
    rgb_bands=("red", "green", "blue"),
    default_bands=("red", "green", "blue"),
    split_sizes=SplitSizes(train=16200, val=5400, test=5400),
    source=TorchGeoSource("EuroSAT", root="data/eurosat"),
    geography=GeographySpec(alias_of="m-eurosat"),

    # Raw EuroSAT pixel statistics, separate from the GeoBench V1 subset's statistics.
    bands=(
        BandSpec("s2", "coastal_aerosol", "B01", mean=1354.41, std=245.718, min=816, max=17720, wavelength_um=0.443),
        BandSpec("s2", "blue", "B02", mean=1118.24, std=333.009, min=0, max=28000, wavelength_um=0.49),
        BandSpec("s2", "green", "B03", mean=1042.93, std=395.094, min=0, max=28000, wavelength_um=0.56),
        BandSpec("s2", "red", "B04", mean=947.627, std=593.752, min=0, max=28000, wavelength_um=0.665),
        BandSpec("s2", "red_edge_1", "B05", mean=1199.47, std=566.418, min=174, max=23381, wavelength_um=0.705),
        BandSpec("s2", "red_edge_2", "B06", mean=1999.79, std=861.185, min=153, max=27791, wavelength_um=0.74),
        BandSpec("s2", "red_edge_3", "B07", mean=2369.22, std=1086.63, min=128, max=28001, wavelength_um=0.783),
        BandSpec("s2", "nir", "B08", mean=2296.83, std=1117.98, min=0, max=28002, wavelength_um=0.842),
        BandSpec("s2", "water_vapour", "B09", mean=732.084, std=404.921, min=40, max=15384, wavelength_um=0.945),
        BandSpec("s2", "swir_cirrus", "B10", mean=12.1133, std=4.7759, min=1, max=183, wavelength_um=1.375),
        BandSpec("s2", "swir_1", "B11", mean=1819.01, std=1002.59, min=5, max=24704, wavelength_um=1.61),
        BandSpec("s2", "swir_2", "B12", mean=1118.92, std=761.305, min=1, max=22210, wavelength_um=2.19),
        BandSpec("s2", "red_edge_4", "B8A", mean=2594.14, std=1231.59, min=91, max=28000, wavelength_um=0.865),
    ),
)

# Longitude-based splits share the archive and original frozen band metadata.
SPATIAL_SPEC = replace(
    SPEC,
    name="eurosat-spatial",
    source=TorchGeoSource("EuroSATSpatial", root="data/eurosat", storage_name="eurosat"),
)
