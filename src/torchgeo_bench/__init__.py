"""Public package exports for torchgeo-bench."""

from importlib.metadata import PackageNotFoundError, version

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


def bootstrap_map(*args, **kwargs):
    """Lazy wrapper for :func:`torchgeo_bench.main.bootstrap_map`."""
    from .main import bootstrap_map as _bootstrap_map

    return _bootstrap_map(*args, **kwargs)


def evaluate_knn(*args, **kwargs):
    """Lazy wrapper for :func:`torchgeo_bench.main.evaluate_knn`."""
    from .main import evaluate_knn as _evaluate_knn

    return _evaluate_knn(*args, **kwargs)


def evaluate_logistic(*args, **kwargs):
    """Lazy wrapper for :func:`torchgeo_bench.main.evaluate_logistic`."""
    from .main import evaluate_logistic as _evaluate_logistic

    return _evaluate_logistic(*args, **kwargs)
