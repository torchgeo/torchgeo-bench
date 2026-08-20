"""Coordinate encoders and classification baselines.

The position encoders intentionally contain no learned task head or upstream
checkpoint. The classification priors are small, network-free baselines for
spatial classification. They do not accept images or produce learned
embeddings. Each prior follows the same API::

    estimator.fit(lon, lat, labels)
    probabilities = estimator.predict_proba(lon, lat)

Coordinates are interpreted as longitude and latitude in degrees.  The
distance-based estimators use Euclidean distance in those degree coordinates;
they are useful baselines, not geodesically accurate spatial models.
"""

from abc import ABC, abstractmethod
from math import log, pi
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

from torchgeo_bench.coordbench.models import LocationEncoder

FloatArray = NDArray[np.float64]
LabelArray = NDArray[Any]


def _unit_xyz(lon: np.ndarray, lat: np.ndarray) -> torch.Tensor:
    """Convert finite geographic coordinates in degrees to unit XYZ tensors."""
    if lon.ndim != 1 or lat.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("lon and lat must be one-dimensional arrays with equal length")
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("lon and lat must contain only finite values")
    if ((lat < -90.0) | (lat > 90.0)).any():
        raise ValueError("lat must be in the inclusive range [-90, 90] degrees")

    lon_radians = torch.as_tensor(np.deg2rad(lon), dtype=torch.float32)
    lat_radians = torch.as_tensor(np.deg2rad(lat), dtype=torch.float32)
    cos_lat = torch.cos(lat_radians)
    return torch.stack(
        (
            cos_lat * torch.cos(lon_radians),
            cos_lat * torch.sin(lon_radians),
            torch.sin(lat_radians),
        ),
        dim=1,
    )


class XYZLocationEncoder(LocationEncoder):
    """Return the three-dimensional unit-sphere representation of each point."""

    name = "xyz"

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        return _unit_xyz(lon, lat).numpy()


class NeRFLocationEncoder(LocationEncoder):
    """Return deterministic NeRF-style Fourier features of spherical XYZ.

    For each XYZ coordinate, frequencies ``2**k * pi`` are applied for
    ``k = 0 .. num_frequencies - 1``. The encoder emits only the transformed
    features by default.
    Set ``include_xyz`` to add raw spherical coordinates as an ablation.

    Args:
        num_frequencies: Number of geometric frequency bands. Must be positive.
        include_xyz: Include the three untransformed XYZ coordinates.
    """

    name = "nerf"

    def __init__(
        self,
        num_frequencies: int = 10,
        include_xyz: bool = False,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.num_frequencies = int(num_frequencies)
        self.include_xyz = bool(include_xyz)

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        xyz = _unit_xyz(lon, lat)
        frequencies = torch.pow(2.0, torch.arange(self.num_frequencies, dtype=torch.float32))
        angles = xyz[:, :, None] * frequencies[None, None, :] * torch.pi
        features = [torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)]
        if self.include_xyz:
            features.insert(0, xyz)
        return torch.cat(features, dim=1).numpy()


class SphericalHarmonicLocationEncoder(LocationEncoder):
    """Return normalized real spherical-harmonic features through degree 3.

    The output has ``(degree + 1)**2`` columns, ordered by increasing degree
    and then the conventional real ``m`` order from ``-l`` to ``l``. Degree
    three is a deliberate upper bound: it provides a compact continuous
    spherical basis without pretending to reproduce a learned task head.

    Args:
        degree: Maximum harmonic degree, from 0 through 3.
    """

    name = "spherical-harmonics"

    def __init__(
        self,
        degree: int = 3,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if degree < 0 or degree > 3:
            raise ValueError("degree must be between 0 and 3")
        self.degree = int(degree)

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        x, y, z = _unit_xyz(lon, lat).unbind(dim=1)
        one = torch.ones_like(x)
        features = [0.28209479177387814 * one]
        if self.degree >= 1:
            features.extend(
                (-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x)
            )
        if self.degree >= 2:
            features.extend(
                (
                    1.0925484305920792 * x * y,
                    -1.0925484305920792 * y * z,
                    0.31539156525252005 * (3.0 * z.square() - 1.0),
                    -1.0925484305920792 * x * z,
                    0.5462742152960396 * (x.square() - y.square()),
                )
            )
        if self.degree >= 3:
            features.extend(
                (
                    -0.5900435899266435 * y * (3.0 * x.square() - y.square()),
                    2.890611442640554 * x * y * z,
                    -0.4570457994644658 * y * (5.0 * z.square() - 1.0),
                    0.3731763325901154 * z * (5.0 * z.square() - 3.0),
                    -0.4570457994644658 * x * (5.0 * z.square() - 1.0),
                    1.445305721320277 * z * (x.square() - y.square()),
                    -0.5900435899266435 * x * (x.square() - 3.0 * y.square()),
                )
            )
        return torch.stack(features, dim=1).numpy()


def _coordinates(lon: ArrayLike, lat: ArrayLike) -> FloatArray:
    """Validate and stack longitude/latitude inputs."""
    longitude = np.asarray(lon, dtype=np.float64)
    latitude = np.asarray(lat, dtype=np.float64)
    if longitude.ndim != 1 or latitude.ndim != 1:
        raise ValueError("lon and lat must be one-dimensional arrays")
    if longitude.shape != latitude.shape:
        raise ValueError("lon and lat must have the same shape")
    if not np.isfinite(longitude).all() or not np.isfinite(latitude).all():
        raise ValueError("lon and lat must contain only finite values")
    return np.column_stack((longitude, latitude))


def _labels(labels: ArrayLike, n_samples: int) -> tuple[LabelArray, LabelArray]:
    """Validate labels and return labels plus their unique class values."""
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if len(values) != n_samples:
        raise ValueError("labels must have the same length as lon and lat")
    if n_samples == 0:
        raise ValueError("fit requires at least one sample")
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise ValueError("labels must not contain NaN or infinite values")
    if values.dtype.kind == "O" and any(value is None for value in values):
        raise ValueError("labels must not contain None")
    try:
        classes = np.unique(values)
    except TypeError as exc:
        raise ValueError("labels must contain comparable scalar class values") from exc
    if classes.ndim != 1 or len(classes) == 0:
        raise ValueError("labels must contain at least one class")
    return values, classes


class SpatialPrior(ABC):
    """Base class for coordinate-only multiclass probability estimators."""

    classes_: LabelArray

    def fit(self, lon: ArrayLike, lat: ArrayLike, labels: ArrayLike) -> Self:
        """Fit the prior from point coordinates and categorical labels."""
        coordinates = _coordinates(lon, lat)
        values, classes = _labels(labels, len(coordinates))
        self._coordinates = coordinates
        self._labels = values
        self.classes_ = classes
        self._fit()
        return self

    @abstractmethod
    def _fit(self) -> None:
        """Fit estimator-specific state after common validation."""

    def predict_proba(self, lon: ArrayLike, lat: ArrayLike) -> FloatArray:
        """Return a ``(n_samples, n_classes)`` array of class probabilities."""
        if not hasattr(self, "classes_"):
            raise ValueError("estimator must be fitted before predict_proba")
        coordinates = _coordinates(lon, lat)
        probabilities = np.asarray(self._predict_proba(coordinates), dtype=np.float64)
        expected = (len(coordinates), len(self.classes_))
        if probabilities.shape != expected:
            raise RuntimeError(
                f"estimator returned shape {probabilities.shape}, expected {expected}"
            )
        if not np.isfinite(probabilities).all() or (probabilities < 0).any():
            raise RuntimeError("estimator returned invalid probabilities")
        row_sums = probabilities.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise RuntimeError("estimator probabilities must sum to one")
        return probabilities

    @abstractmethod
    def _predict_proba(self, coordinates: FloatArray) -> FloatArray:
        """Predict probabilities for validated coordinates."""


class UniformPrior(SpatialPrior):
    """Predict every fitted class with equal probability."""

    def _fit(self) -> None:
        pass

    def _predict_proba(self, coordinates: FloatArray) -> FloatArray:
        return np.full(
            (len(coordinates), len(self.classes_)), 1.0 / len(self.classes_), dtype=np.float64
        )


class ClassFrequencyPrior(SpatialPrior):
    """Predict the empirical global class-frequency distribution."""

    def _fit(self) -> None:
        self.prior_ = np.array([(self._labels == cls).mean() for cls in self.classes_])

    def _predict_proba(self, coordinates: FloatArray) -> FloatArray:
        return np.broadcast_to(self.prior_, (len(coordinates), len(self.classes_))).copy()


class GridPrior(SpatialPrior):
    """Predict empirical class frequencies within longitude/latitude grid cells.

    Args:
        cell_size: Cell width in degrees.  A scalar uses square cells; a
            ``(longitude, latitude)`` pair permits rectangular cells.
        smoothing: Non-negative pseudocount added to every class in every
            occupied cell.  Zero preserves raw empirical frequencies.
    """

    def __init__(
        self, cell_size: float | tuple[float, float] = 10.0, smoothing: float = 0.0
    ) -> None:
        if isinstance(cell_size, tuple):
            sizes = np.asarray(cell_size, dtype=np.float64)
        else:
            sizes = np.full(2, cell_size, dtype=np.float64)
        if sizes.shape != (2,) or not np.isfinite(sizes).all() or (sizes <= 0).any():
            raise ValueError("cell_size must contain two positive finite values")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be a non-negative finite value")
        self.cell_size = tuple(float(size) for size in sizes)
        self.smoothing = float(smoothing)

    def _cell_ids(self, coordinates: FloatArray) -> NDArray[np.int64]:
        return np.floor(coordinates / self.cell_size).astype(np.int64)

    def _fit(self) -> None:
        cell_ids = self._cell_ids(self._coordinates)
        self._cell_probabilities: dict[tuple[int, int], FloatArray] = {}
        for cell in map(tuple, np.unique(cell_ids, axis=0)):
            mask = (cell_ids[:, 0] == cell[0]) & (cell_ids[:, 1] == cell[1])
            counts = np.array(
                [(self._labels[mask] == cls).sum() for cls in self.classes_], dtype=np.float64
            )
            counts += self.smoothing
            self._cell_probabilities[cell] = counts / counts.sum()
        self._global = np.array(
            [(self._labels == cls).mean() for cls in self.classes_], dtype=np.float64
        )

    def _predict_proba(self, coordinates: FloatArray) -> FloatArray:
        result = np.empty((len(coordinates), len(self.classes_)), dtype=np.float64)
        for index, cell_array in enumerate(self._cell_ids(coordinates)):
            result[index] = self._cell_probabilities.get(tuple(cell_array), self._global)
        return result


class NearestNeighborPrior(SpatialPrior):
    """Predict from labels of the nearest fitted points.

    ``weights="distance"`` gives closer points more influence and uses only
    exact-coordinate matches when one or more neighbors have zero distance.
    "distance" is Euclidean distance in degree coordinates, not a great-circle
    distance.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform") -> None:
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, (int, np.integer)):
            raise ValueError("n_neighbors must be a positive integer")
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be a positive integer")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        self.n_neighbors = int(n_neighbors)
        self.weights = weights

    def _fit(self) -> None:
        self._coordinates_tensor = torch.as_tensor(self._coordinates, dtype=torch.float64)
        self._encoded_labels = torch.as_tensor(
            np.searchsorted(self.classes_, self._labels), dtype=torch.int64
        )

    def _predict_proba(self, coordinates: FloatArray) -> FloatArray:
        with torch.no_grad():
            query = torch.as_tensor(coordinates, dtype=torch.float64)
            n_neighbors = min(self.n_neighbors, len(self._coordinates_tensor))
            distances, indices = torch.cdist(query, self._coordinates_tensor).topk(
                n_neighbors, largest=False, sorted=True
            )
            if self.weights == "uniform":
                weights = torch.ones_like(distances)
            else:
                exact = distances == 0
                weights = torch.where(exact, torch.zeros_like(distances), distances.reciprocal())
                exact_rows = exact.any(dim=1)
                weights[exact_rows] = exact[exact_rows].to(weights.dtype)

            result = torch.zeros((len(coordinates), len(self.classes_)), dtype=torch.float64)
            class_indices = self._encoded_labels[indices]
            result.scatter_add_(1, class_indices, weights)
            result /= result.sum(dim=1, keepdim=True)
            return result.numpy()


class KDEPrior(SpatialPrior):
    """Predict with class-conditional Gaussian KDEs and empirical class priors.

    The bandwidth is measured in degrees and the KDE uses Euclidean lon/lat
    geometry.  Classes with one sample are valid but yield a very local KDE.
    """

    def __init__(self, bandwidth: float = 10.0, smoothing: float = 0.0) -> None:
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be a positive finite value")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be a non-negative finite value")
        self.bandwidth = float(bandwidth)
        self.smoothing = float(smoothing)

    def _fit(self) -> None:
        self._class_coordinates: list[torch.Tensor] = []
        counts: list[int] = []
        for cls in self.classes_:
            points = self._coordinates[self._labels == cls]
            counts.append(len(points))
            self._class_coordinates.append(torch.as_tensor(points, dtype=torch.float64))
        counts_array = np.asarray(counts, dtype=np.float64) + self.smoothing
        self._log_prior = torch.log(torch.as_tensor(counts_array / counts_array.sum()))

    def _predict_proba(self, coordinates: FloatArray) -> FloatArray:
        with torch.no_grad():
            query = torch.as_tensor(coordinates, dtype=torch.float64)
            log_probabilities = []
            normalizer = log(2.0 * pi * self.bandwidth**2)
            for points in self._class_coordinates:
                squared_distances = torch.cdist(query, points).square()
                log_density = (-0.5 * squared_distances / self.bandwidth**2).logsumexp(dim=1)
                log_density -= log(len(points)) + normalizer
                log_probabilities.append(log_density)
            log_probabilities = torch.stack(log_probabilities, dim=1) + self._log_prior
            return torch.softmax(log_probabilities, dim=1).numpy()


# Descriptive aliases make the intended baseline mapping easy to discover.
EmpiricalPrior = ClassFrequencyPrior
DistancePrior = NearestNeighborPrior
UniformBaseline = UniformPrior
FrequencyBaseline = ClassFrequencyPrior
GridBaseline = GridPrior
NearestNeighborBaseline = NearestNeighborPrior
KDEBaseline = KDEPrior

__all__ = [
    "XYZLocationEncoder",
    "NeRFLocationEncoder",
    "SphericalHarmonicLocationEncoder",
    "SpatialPrior",
    "UniformPrior",
    "ClassFrequencyPrior",
    "EmpiricalPrior",
    "GridPrior",
    "NearestNeighborPrior",
    "DistancePrior",
    "KDEPrior",
    "UniformBaseline",
    "FrequencyBaseline",
    "GridBaseline",
    "NearestNeighborBaseline",
    "KDEBaseline",
]
