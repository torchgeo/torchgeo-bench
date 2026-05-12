"""Benchmark dataset registry for torchgeo-bench.

Public API
----------
.. autofunction:: get_datasets
.. autofunction:: get_bench_dataset_class
.. autofunction:: list_datasets
.. autoclass:: BandSpec
.. autoclass:: BenchDataset
"""

from .base import BandSpec, BenchDataset
from .benv2 import BENV2
from .burn_scars import BurnScars
from .caffe import CaFFe
from .cloudsen12 import CloudSEN12
from .dynamic_earthnet import DynamicEarthNet
from .eurosat import EuroSAT, EuroSATSpatial
from .flair2 import FLAIR2
from .forestnet import Forestnet
from .fotw import FieldsOfTheWorld
from .kuro_siwo import KuroSiwo
from .loading import (
    get_bench_dataset_class,
    get_datasets,
    list_datasets,
)
from .m_bigearthnet import MBigEarthNet
from .m_brick_kiln import MBrickKiln
from .m_eurosat import MEurosat
from .m_forestnet import MForestnet
from .m_pv4ger import MPv4ger
from .m_so2sat import MSo2Sat
from .pastis import PASTIS
from .so2sat import So2Sat
from .spacenet2 import SpaceNet2
from .spacenet7 import SpaceNet7
from .treesatai import TreeSatAI

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
