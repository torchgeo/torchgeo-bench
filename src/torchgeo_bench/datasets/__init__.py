"""Benchmark dataset registry for torchgeo-bench.

Individual dataset classes load lazily (via module ``__getattr__``, mirroring
:mod:`torchgeo_bench`) so that ``import torchgeo_bench.datasets`` — and
therefore CLI startup — stays fast.

Public API
----------
.. autofunction:: get_datasets
.. autofunction:: get_bench_dataset_class
.. autofunction:: list_datasets
.. autoclass:: BandSpec
.. autoclass:: BenchDataset
"""

from importlib import import_module

from .base import BandSpec, BenchDataset
from .loading import (
    get_bench_dataset_class,
    get_datasets,
    list_datasets,
)

__all__ = [
    # Core types
    "BandSpec",
    "BenchDataset",
    # Loading API
    "get_bench_dataset_class",
    "get_datasets",
    "list_datasets",
    # Individual dataset classes
    "BENV2",
    "BurnScars",
    "CaFFe",
    "CloudSEN12",
    "DynamicEarthNet",
    "EuroSAT",
    "EuroSATSpatial",
    "FLAIR2",
    "Forestnet",
    "FieldsOfTheWorld",
    "KuroSiwo",
    "MBigEarthNet",
    "MBrickKiln",
    "MEurosat",
    "MForestnet",
    "MPv4ger",
    "MSo2Sat",
    "PASTIS",
    "So2Sat",
    "SpaceNet2",
    "SpaceNet7",
    "TreeSatAI",
]

# Class name -> defining submodule, resolved on first attribute access.
_LAZY_CLASSES: dict[str, str] = {
    "BENV2": "benv2",
    "BurnScars": "burn_scars",
    "CaFFe": "caffe",
    "CloudSEN12": "cloudsen12",
    "DynamicEarthNet": "dynamic_earthnet",
    "EuroSAT": "eurosat",
    "EuroSATSpatial": "eurosat",
    "FLAIR2": "flair2",
    "Forestnet": "forestnet",
    "FieldsOfTheWorld": "fotw",
    "KuroSiwo": "kuro_siwo",
    "MBigEarthNet": "m_bigearthnet",
    "MBrickKiln": "m_brick_kiln",
    "MEurosat": "m_eurosat",
    "MForestnet": "m_forestnet",
    "MPv4ger": "m_pv4ger",
    "MSo2Sat": "m_so2sat",
    "PASTIS": "pastis",
    "So2Sat": "so2sat",
    "SpaceNet2": "spacenet2",
    "SpaceNet7": "spacenet7",
    "TreeSatAI": "treesatai",
}


def __getattr__(name: str) -> object:
    if name in _LAZY_CLASSES:
        cls = getattr(import_module(f".{_LAZY_CLASSES[name]}", __name__), name)
        globals()[name] = cls  # cache so subsequent lookups skip __getattr__
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_CLASSES))
