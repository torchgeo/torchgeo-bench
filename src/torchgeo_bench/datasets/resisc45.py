"""NWPU-RESISC45 (torchgeo) benchmark dataset.

RESISC45 is one of the three most-evaluated benchmarks in the geospatial
foundation-model literature, and the single most divergent: the same
released Scale-MAE ViT-L checkpoint is reported at 33.0 and 89.6 linear-probe
accuracy by different papers under the same nominal protocol.  Running it
under one harness is the point of having it here.

This module is also the reference example for wrapping a torchgeo dataset
whose loader does **not** accept a ``bands`` argument.  :class:`EuroSAT`
forwards band codes straight to torchgeo; RESISC45 is a three-channel JPEG
:class:`~torchvision.datasets.ImageFolder`, so channel selection has to
happen in the wrapper.  See :meth:`RESISC45.get_dataset`.
"""

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchgeo.datasets import RESISC45 as TGRESISC45
from torchvision.transforms import Compose

from .base import BandSpec, BenchDataset


class RESISC45(BenchDataset):
    """Aerial scene classification, 45 classes, via torchgeo.

    31,500 RGB images (700 per class) at 256x256, extracted from Google Earth
    by Northwestern Polytechnical University.  Splits are torchgeo's published
    60/20/20 partition (18,900 / 6,300 / 6,300).

    The imagery is 8-bit RGB with no radiometric calibration and no per-image
    geolocation -- it is a curated scene-recognition set, not a sensor
    product.  Band metadata below reflects that: the ``aerial`` sensor tag
    and nominal visible-light wavelengths are descriptive, not measured.
    RESISC45 spans 0.2--30 m/px with no per-image scale metadata, so the
    ``aerial`` tag's fixed GSD is a deliberate approximation, not a claim of
    true fixed resolution -- chosen so resolution-aware / sensor-routed
    models (OlmoEarth, UniverSat, ...) treat it as ordinary RGB aerial
    imagery instead of rejecting or silently mis-routing it.
    """

    name = "resisc45"
    task = "classification"
    num_classes = 45
    multilabel = False
    rgb_bands = ["red", "green", "blue"]
    split_sizes = {"train": 18900, "val": 6300, "test": 6300}
    supports_partitions = False

    # Statistics computed over the 18,900-image train split in raw 0-255 units
    # (scripts/compute_band_statistics.py).  ``source_name`` is the channel
    # position in the RGB JPEG -- unlike a multispectral product there is no
    # band key in the file to name.  Wavelengths are nominal visible-light
    # centres: Google Earth composites many sensors, so no single response
    # curve applies.  Sensor tag is ``aerial`` (1 m nominal GSD, same
    # modality table entry as NAIP) -- an approximation, since RESISC45
    # actually spans 0.2--30 m/px with no per-image scale metadata, but this
    # keeps sensor-routed models (OlmoEarth, UniverSat) working instead of
    # rejecting the dataset outright.
    # fmt: off
    bands = [
        BandSpec("aerial", "red", "R", mean=93.8939, std=51.8492, min=0, max=255, wavelength_um=0.65),
        BandSpec("aerial", "green", "G", mean=97.1123, std=47.2366, min=0, max=255, wavelength_um=0.55),
        BandSpec("aerial", "blue", "B", mean=87.5678, std=47.0631, min=0, max=255, wavelength_um=0.45),
    ]
    # fmt: on

    @classmethod
    def data_root(cls) -> Path:
        """Return ``Path("data/resisc45")``; torchgeo manages the layout below."""
        return Path("data/resisc45")

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return the wrapped torchgeo dataset for the split.

        ``torchgeo.datasets.RESISC45`` always yields all three channels, so
        any band subset is applied here as a sample transform rather than
        pushed down to the loader.  The selection runs *before* the caller's
        ``transform`` (the resize built by
        :func:`~torchgeo_bench.datasets.get_datasets`) so the resize
        only touches channels that survive.
        """
        del partition
        specs = self.select_band_specs(bands)
        indices = [self.bands.index(spec) for spec in specs]
        select = _make_band_select(indices, len(self.bands))
        if select is not None:
            transform = select if transform is None else Compose([select, transform])
        return TGRESISC45(
            root=str(self.data_root()),
            split=split,
            transforms=transform,
        )


def _make_band_select(indices: list[int], n_bands: int) -> Callable[[dict], dict] | None:
    """Return a transform selecting ``indices`` from the channel axis.

    ``None`` when the selection is the identity (every band, in order), so the
    common ``bands="all"`` and ``bands="rgb"`` paths add no per-sample work.
    """
    if indices == list(range(n_bands)):
        return None
    index = torch.tensor(indices)

    def _select(sample: dict) -> dict:
        sample["image"] = sample["image"].index_select(-3, index)
        return sample

    return _select
