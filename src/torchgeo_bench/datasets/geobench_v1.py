"""Construct a local GeoBench V1 JSON-shard split from its resolved input."""

from collections.abc import Callable
from typing import Literal

from torch.utils.data import Dataset

from ._v1_webdataset import GeoBenchv1Sharded
from .input import ResolvedInput, Split
from .spec import DatasetSpec, V1Source


def load_v1_split(
    spec: DatasetSpec,
    split: Split,
    *,
    inputs: ResolvedInput,
    partition: str = "default",
    transform: Callable | None = None,
) -> Dataset:
    """Load exactly one local JSON-shard split, without acquiring data."""
    source = spec.source
    assert isinstance(source, V1Source)
    v1_split: Literal["train", "valid", "test"] = (
        source.validation_split if split == "val" else split
    )
    return GeoBenchv1Sharded(
        root=source.root,
        dataset_name=spec.storage_name,
        split=v1_split,
        partition=partition,
        bands=tuple(band.source_name for band in inputs.bands),
        transform=transform,
    )
