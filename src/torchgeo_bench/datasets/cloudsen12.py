"""CloudSEN12 (GeoBench V2) benchmark dataset."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from torch.utils.data import Dataset

from .base import BandSpec, BenchDataset


class CloudSEN12(BenchDataset):
    """Sentinel-2 cloud segmentation (4 classes)."""

    name = "cloudsen12"
    task = "segmentation"
    num_classes = 4
    multilabel = False
    rgb_bands = ["b04", "b03", "b02"]
    split_sizes = {"train": 4000, "val": 535, "test": 975}

    bands = [
        BandSpec("s2", "b01", "B01", wavelength_um=0.443, mean=2030.2444, std=2723.436),
        BandSpec("s2", "b02", "B02", wavelength_um=0.49, mean=2074.8171, std=2691.3027),
        BandSpec("s2", "b03", "B03", wavelength_um=0.56, mean=2209.8074, std=2539.9136),
        BandSpec("s2", "b04", "B04", wavelength_um=0.665, mean=2247.9275, std=2538.5208),
        BandSpec("s2", "b05", "B05", wavelength_um=0.705, mean=2589.5935, std=2504.3284),
        BandSpec("s2", "b06", "B06", wavelength_um=0.74, mean=3103.5212, std=2241.7446),
        BandSpec("s2", "b07", "B07", wavelength_um=0.783, mean=3277.9094, std=2145.6677),
        BandSpec("s2", "b08", "B08", wavelength_um=0.842, mean=3331.6318, std=2176.9978),
        BandSpec("s2", "b8a", "B8A", wavelength_um=0.865, mean=3377.5447, std=2066.7637),
        BandSpec("s2", "b09", "B09", wavelength_um=0.945, mean=4038.1931, std=3083.1799),
        BandSpec("s2", "b11", "B11", wavelength_um=1.61, mean=2448.748, std=1595.0652),
        BandSpec("s2", "b12", "B12", wavelength_um=2.19, mean=1907.7285, std=1474.1177),
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

        return gb_v2.GeoBenchCloudSen12(
            root=os.path.join(self.root, self.name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
