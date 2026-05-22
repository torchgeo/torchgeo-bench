"""Benchmark dataset registry for torchgeo-bench.

Public API
----------
.. autofunction:: get_datasets
.. autofunction:: get_bench_dataset_class
.. autofunction:: list_datasets
.. autoclass:: BandSpec
.. autoclass:: BenchDataset
"""

from .advance import ADVANCE
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
from .resisc45 import RESISC45
from .sen12ms_cr import SEN12MS, SEN12MSCRC1, SEN12MSCRC2, SEN12MSCRC3, SEN12MSCRC4
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
    "ADVANCE",
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
    "RESISC45",
    "SEN12MS",
    "SEN12MSCRC1",
    "SEN12MSCRC2",
    "SEN12MSCRC3",
    "SEN12MSCRC4",
    "So2Sat",
    "SpaceNet2",
    "SpaceNet7",
    "TreeSatAI",
]
