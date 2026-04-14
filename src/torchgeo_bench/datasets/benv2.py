"""BENV2 (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class BENV2(BenchDataset):
    """Sentinel-2 + SAR multi-class classification (19 classes).

    BigEarthNet V2 with 12 Sentinel-2 optical bands and 2 SAR bands.
    """

    name = "benv2"
    task = "classification"
    num_classes = 19
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 20000, "val": 4000, "test": 4000}

    bands = [
        BandSpec("s2", "b01", "B01", wavelength_um=0.443, mean=355.962, std=512.342),
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=414.3731, std=541.9492),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=594.0964, std=532.5798),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=559.0434, std=607.0201),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=919.41, std=646.3411),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=1794.6605, std=1041.3501),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=2091.4595, std=1231.7878),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=2241.5178, std=1340.4662),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=2288.0303, std=1316.0288),
        BandSpec("s2", "b09", "B09", wavelength_um=0.945, mean=2289.5381, std=1267.3955),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=1556.9587, std=984.2933),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=973.8273, std=753.2082),
        BandSpec("sar", "vv", "VV", mean=-18.9633, std=5.3961),
        BandSpec("sar", "vh", "VH", mean=-12.0919, std=4.5749),
    ]

    def __init__(self, root: str | Path | None = None) -> None:
        if root is None:
            root = os.getenv("GEOBENCH_V2_ROOT", "data/geobenchv2")
        super().__init__(root)

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
        normalize: str = "mean_stdev",
    ) -> Dataset:
        """Return a PyTorch Dataset for the given split."""
        del partition, normalize
        import geobench_v2.datasets as gb_v2

        return gb_v2.GeoBenchBENV2(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
