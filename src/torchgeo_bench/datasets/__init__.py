"""Lightweight dataset definitions and single-split loading."""

from torchgeo_bench.bands import BandSpec

from .catalog import get_dataset_spec, get_dataset_task, list_datasets, list_v2_datasets
from .input import LoadedSplit, ResolvedInput
from .loading import load_split
from .spec import (
    DatasetCapabilities,
    DatasetSpec,
    GeographySpec,
    SplitSizes,
    TorchGeoSource,
    V1Source,
    V2Source,
)

__all__ = [
    "BandSpec",
    "DatasetCapabilities",
    "DatasetSpec",
    "GeographySpec",
    "LoadedSplit",
    "ResolvedInput",
    "SplitSizes",
    "TorchGeoSource",
    "V1Source",
    "V2Source",
    "get_dataset_spec",
    "get_dataset_task",
    "list_datasets",
    "list_v2_datasets",
    "load_split",
]
