"""EuroSAT (torchgeo) benchmark dataset template.

Demonstrates how to wrap a non-GeoBench :class:`~torch.utils.data.Dataset`
in a :class:`~torchgeo_bench.datasets.base.BenchDataset`.  The data and
splits come from :class:`torchgeo.datasets.EuroSAT`; metadata and the
``BenchDataset`` interface live here.
"""

from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset
from torchgeo.datasets import EuroSAT as TGEuroSAT
from torchgeo.datasets import EuroSATSpatial as TGEuroSATSpatial

from .base import BandSpec, BenchDataset


class EuroSAT(BenchDataset):
    """Sentinel-2 land-use classification (10 classes), via torchgeo.

    13 Sentinel-2 spectral bands. Identical task and class set as
    :class:`~torchgeo_bench.datasets.MEurosat` (GeoBench V1) but loads
    data through :class:`torchgeo.datasets.EuroSAT`, so file layout and
    download behaviour are managed by torchgeo.
    """

    name = "eurosat"
    task = "classification"
    num_classes = 10
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 16200, "val": 5400, "test": 5400}
    supports_partitions = False

    # Band statistics mirror m-eurosat (computed from the same EuroSAT data).
    # fmt: off
    bands = [
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
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/eurosat")`` (torchgeo manages its own layout below)."""
        return Path("data/eurosat")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        normalize: str = "mean_stdev",
    ) -> Dataset:
        """Return a :class:`torchgeo.datasets.EuroSAT` for the given split."""
        del partition, normalize
        band_codes = tuple(spec.source_name for spec in self.select_band_specs(bands))
        return TGEuroSAT(
            root=str(self.data_root()),
            split=split,
            bands=band_codes,
            transforms=transform,
        )


class EuroSATSpatial(EuroSAT):
    """EuroSAT with longitude-based 60/20/20 train/val/test splits.

    Uses :class:`torchgeo.datasets.EuroSATSpatial`, which partitions tiles
    by longitude so train/val/test regions are spatially disjoint. Same
    27000 images, classes, bands, and stats as :class:`EuroSAT`; only the
    split assignment differs. Stronger generalization signal than the
    default random split.
    """

    name = "eurosat-spatial"
    # Longitude-based 60/20/20: same totals as the random split, just
    # reassigned across regions.
    split_sizes = {"train": 16200, "val": 5400, "test": 5400}

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/eurosat")`` — shares the archive with :class:`EuroSAT`."""
        # Both classes use the same EuroSATallBands.zip; only the split
        # txt files differ. Sharing the root avoids a second 2GB download.
        return Path("data/eurosat")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        normalize: str = "mean_stdev",
    ) -> Dataset:
        """Return a :class:`torchgeo.datasets.EuroSATSpatial` for the given split."""
        del partition, normalize
        band_codes = tuple(spec.source_name for spec in self.select_band_specs(bands))
        return TGEuroSATSpatial(
            root=str(self.data_root()),
            split=split,
            bands=band_codes,
            transforms=transform,
        )
