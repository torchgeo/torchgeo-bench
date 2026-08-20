"""Classification baselines for coordinate-only tasks.

The priors are small, network-free baselines for spatial classification. They
use training coordinates and labels, then return class probabilities for new
coordinates. Coordinates are interpreted as longitude and latitude in degrees;
distance-based estimators use Euclidean distance in those degree coordinates.
"""

from abc import ABC, abstractmethod
from math import log, pi
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
LabelArray = NDArray[Any]


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
