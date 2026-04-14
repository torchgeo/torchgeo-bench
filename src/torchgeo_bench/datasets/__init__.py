"""Benchmark dataset registry for torchgeo-bench.

Public API
----------
.. autofunction:: get_bench_dataset
.. autofunction:: list_datasets
.. autoclass:: BandSpec
.. autoclass:: BenchDataset
"""

from __future__ import annotations

# Backward-compatible re-exports from the legacy datasets module.
# These allow existing code like ``from torchgeo_bench.datasets import get_datasets``
# to continue working during migration.
from ._legacy import (
    DEFAULT_GEOBENCH_ROOT,
    DEFAULT_GEOBENCH_V2_ROOT,
    NUM_CLASSES_PER_DATASET,
    PARTITION_NAMES,
    V2_DATASETS,
    V2_TASK_TYPES,
    get_datasets,
    is_dataset_available,
)
from .base import BandSpec, BenchDataset
from .benv2 import BENV2
from .burn_scars import BurnScars
from .caffe import CaFFe
from .cloudsen12 import CloudSEN12
from .dynamic_earthnet import DynamicEarthNet
from .flair2 import FLAIR2
from .forestnet import Forestnet
from .fotw import FieldsOfTheWorld
from .kuro_siwo import KuroSiwo
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

_REGISTRY: dict[str, type[BenchDataset]] = {
    cls.name: cls
    for cls in [
        # V1 classification
        MEurosat,
        MForestnet,
        MSo2Sat,
        MPv4ger,
        MBrickKiln,
        MBigEarthNet,
        # V2 classification
        BENV2,
        TreeSatAI,
        So2Sat,
        Forestnet,
        CaFFe,
        # V2 segmentation
        BurnScars,
        CloudSEN12,
        DynamicEarthNet,
        FLAIR2,
        FieldsOfTheWorld,
        KuroSiwo,
        PASTIS,
        SpaceNet2,
        SpaceNet7,
    ]
}


def get_bench_dataset(name: str, root: str | None = None) -> BenchDataset:
    """Look up a dataset by name and return an instance.

    Args:
        name: Dataset identifier (e.g. ``"m-eurosat"``, ``"burn_scars"``).
        root: Optional root directory.  If ``None``, the dataset class
            resolves the root from environment variables.

    Returns:
        An instantiated :class:`BenchDataset` subclass.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    cls = _REGISTRY[name]
    return cls(root=root) if root is not None else cls()


def list_datasets() -> list[str]:
    """Return sorted names of all registered benchmark datasets."""
    return sorted(_REGISTRY)


__all__ = [
    "BandSpec",
    "BenchDataset",
    "get_bench_dataset",
    "list_datasets",
    # Individual dataset classes
    "BENV2",
    "BurnScars",
    "CaFFe",
    "CloudSEN12",
    "DynamicEarthNet",
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
    # Legacy re-exports (backward compatibility)
    "DEFAULT_GEOBENCH_ROOT",
    "DEFAULT_GEOBENCH_V2_ROOT",
    "NUM_CLASSES_PER_DATASET",
    "PARTITION_NAMES",
    "V2_DATASETS",
    "V2_TASK_TYPES",
    "get_datasets",
    "is_dataset_available",
]
