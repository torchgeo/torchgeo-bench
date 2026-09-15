"""One read-only catalog derived from lightweight per-dataset definitions."""

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Literal

from . import (
    benv2,
    burn_scars,
    caffe,
    cloudsen12,
    dynamic_earthnet,
    eurosat,
    flair2,
    forestnet,
    fotw,
    kuro_siwo,
    m_bigearthnet,
    m_brick_kiln,
    m_eurosat,
    m_forestnet,
    m_pv4ger,
    m_so2sat,
    pastis,
    resisc45,
    so2sat,
    spacenet2,
    spacenet7,
    treesatai,
)
from .spec import DatasetSpec, Task


def _make_catalog(specs: Iterable[DatasetSpec]) -> Mapping[str, DatasetSpec]:
    catalog = {}
    for spec in specs:
        if spec.name in catalog:
            raise ValueError(f"Duplicate dataset definition: {spec.name!r}")
        catalog[spec.name] = spec
    return MappingProxyType(catalog)


_CATALOG = _make_catalog(
    (
        benv2.SPEC,
        burn_scars.SPEC,
        caffe.SPEC,
        cloudsen12.SPEC,
        dynamic_earthnet.SPEC,
        eurosat.SPEC,
        eurosat.SPATIAL_SPEC,
        flair2.SPEC,
        forestnet.SPEC,
        fotw.SPEC,
        kuro_siwo.SPEC,
        m_bigearthnet.SPEC,
        m_brick_kiln.SPEC,
        m_eurosat.SPEC,
        m_forestnet.SPEC,
        m_pv4ger.SPEC,
        m_so2sat.SPEC,
        pastis.SPEC,
        resisc45.SPEC,
        so2sat.SPEC,
        spacenet2.SPEC,
        spacenet7.SPEC,
        treesatai.SPEC,
    )
)


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the complete immutable definition, without importing any readers."""
    if name not in _CATALOG:
        available = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _CATALOG[name]


def list_datasets(*, source: Literal["v1", "v2", "torchgeo"] | None = None) -> list[str]:
    """Return sorted canonical identities, optionally filtered by source family."""
    return sorted(
        name for name, spec in _CATALOG.items() if source is None or spec.source.kind == source
    )


def list_v2_datasets() -> list[str]:
    """Return all V2 identities, including classification and segmentation."""
    return list_datasets(source="v2")


def get_dataset_task(name: str) -> Task:
    """Return the task declared by the authoritative definition."""
    return get_dataset_spec(name).task


def download_command(dataset: DatasetSpec | str) -> str:
    """Return the explicit acquisition command for a benchmark's source identity."""
    spec = get_dataset_spec(dataset) if isinstance(dataset, str) else dataset
    match spec.source.kind:
        case "v1":
            target = f"geobench_v1 --datasets {spec.storage_name}"
        case "v2":
            target = f"geobench_v2 --datasets {spec.storage_name}"
        case "torchgeo":
            target = spec.storage_name
    return f"torchgeo-bench download {target}"
