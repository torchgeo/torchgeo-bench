"""Minimal custom location encoder for the CoordBench runner.

Run from the repository root:

    PYTHONPATH=. uv run torchgeo-bench run mode=coord model=sincos \
        model._target_=examples.coordbench_location_encoder.FourierLocationEncoder \
        model.name=fourier +model.num_frequencies=8 \
        coord.names=california_housing coord.methods=[linear] coord.folds=2 \
        device=cpu coord.output=results/fourier_coordbench.csv
"""

import numpy as np

from torchgeo_bench.coordbench import LocationEncoder


class FourierLocationEncoder(LocationEncoder):
    """Encode longitude and latitude with multi-frequency sine/cosine features."""

    name = "fourier"

    def __init__(
        self,
        num_frequencies: int = 8,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.frequencies = 2.0 ** np.arange(num_frequencies, dtype=np.float32)

    def _encode(
        self,
        lon: np.ndarray,
        lat: np.ndarray,
        _year: np.ndarray | None,
    ) -> np.ndarray:
        coords = np.deg2rad(np.column_stack((lon, lat))).astype(np.float32)
        angles = coords[:, :, None] * self.frequencies[None, None, :]
        features = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
        return features.reshape(len(lon), -1).astype(np.float32)
