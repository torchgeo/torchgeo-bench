"""Public package exports for torchgeo-bench."""

from importlib.metadata import PackageNotFoundError, version

from .main import bootstrap_map, evaluate_knn, evaluate_logistic

try:
    __version__ = version("torchgeo-bench")
except PackageNotFoundError:  # editable / pre-install fallback
    __version__ = "0.3.0"

__author__ = "torchgeo-bench contributors"

__all__: list[str] = [
    "bootstrap_map",
    "evaluate_knn",
    "evaluate_logistic",
]
