"""Download benchmark datasets into ``data/``.

Targets:

- ``geobench_v1``: JSON-metadata shards under ``<output>/classification_v1.0_wds/``.
- ``geobench_v2``: datasets from ``aialliance/<name>`` under ``<output>/geobenchv2/``.
- ``eurosat`` — torchgeo's EuroSAT downloader, into ``<output>/eurosat``.
- ``resisc45`` — torchgeo's NWPU-RESISC45 downloader, into ``<output>/resisc45``.

V1 uses the pinned ``calebrob6/geobenchv1-webdataset`` mirror.

Use ``--datasets`` to select a GeoBench subset.
"""

import logging
from pathlib import Path

from torchgeo_bench.datasets import (
    DatasetSpec,
    TorchGeoSource,
    V1Source,
    get_dataset_spec,
    list_datasets,
    list_v2_datasets,
)

logger = logging.getLogger(__name__)

GEOBENCH_V2_REPO_PREFIX = "aialliance"

DEFAULT_V2_DATASETS: tuple[str, ...] = tuple(list_v2_datasets())

V1_DATASETS: tuple[str, ...] = tuple(list_datasets(source="v1"))
TORCHGEO_DATASETS: tuple[str, ...] = tuple(list_datasets(source="torchgeo"))
DOWNLOADABLE_DATASETS: tuple[str, ...] = tuple(list_datasets())


def _validate_names(names: list[str]) -> list[str]:
    """Validate and deduplicate names before creating download roots."""
    if not names:
        raise ValueError("datasets must contain at least one dataset name")
    unique = list(dict.fromkeys(names))
    unknown = sorted(set(unique) - set(DOWNLOADABLE_DATASETS))
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(DOWNLOADABLE_DATASETS)}"
        )
    return unique


def download_geobench_v1(output_dir: Path, datasets: list[str] | None = None) -> None:
    """Download verified, pickle-free GeoBench V1 shards to ``output_dir``.

    Args:
        output_dir: Benchmark data root (typically ``data/``).
        datasets: Specific dataset names to fetch from the sharded mirror.
            ``None`` downloads all six classification datasets.

    Raises:
        ValueError: If dataset names are invalid or an archive checksum fails.
    """
    names = list(V1_DATASETS) if datasets is None else list(dict.fromkeys(datasets))
    if not names:
        raise ValueError("datasets must contain at least one GeoBench V1 dataset name.")
    unknown = sorted(set(names) - set(V1_DATASETS))
    if unknown:
        raise ValueError(f"Unknown GeoBench V1 dataset(s): {', '.join(unknown)}")

    from torchgeo_bench._v1_download import download_sharded_root

    root = Path(V1Source().root).relative_to("data")
    download_sharded_root(
        Path(output_dir) / root, [get_dataset_spec(name).storage_name for name in names]
    )


def download_geobench_v2_dataset(name: str, v2_root: Path) -> None:
    """Download a single GeoBench V2 dataset into ``v2_root/<name>``."""
    from huggingface_hub import snapshot_download

    spec = get_dataset_spec(name)
    if spec.source.kind != "v2":
        raise ValueError(f"{name!r} is not a GeoBench V2 dataset")
    target = v2_root / spec.storage_name
    target.mkdir(parents=True, exist_ok=True)
    repo_id = f"{GEOBENCH_V2_REPO_PREFIX}/{spec.storage_name}"
    logger.info("Downloading %s -> %s", repo_id, target)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target,
    )


def download_geobench_v2(output_dir: Path, datasets: list[str] | None = None) -> None:
    """Download GeoBench V2 datasets into ``output_dir/geobenchv2/<name>``.

    Args:
        output_dir: Benchmark data root (typically ``data/``).
        datasets: Specific dataset names to fetch. ``None`` downloads
            :data:`DEFAULT_V2_DATASETS`.
    """
    if datasets is None:
        names = list(DEFAULT_V2_DATASETS)
    elif not datasets:
        raise ValueError("datasets must contain at least one GeoBench V2 dataset name.")
    else:
        names = list(dict.fromkeys(datasets))

    unknown = sorted(set(names) - set(DEFAULT_V2_DATASETS))
    if unknown:
        raise ValueError(
            f"Unknown GeoBench V2 dataset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(DEFAULT_V2_DATASETS)}"
        )
    v2_root = Path(output_dir) / Path(get_dataset_spec(names[0]).source.root).relative_to("data")
    v2_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %d GeoBench v2 dataset(s) to %s", len(names), v2_root)
    for name in names:
        download_geobench_v2_dataset(name, v2_root)
    logger.info("GeoBench v2 download complete.")


def download_eurosat(output_dir: Path) -> None:
    """Download EuroSAT imagery and both standard/spatial split definitions."""
    _download_torchgeo(get_dataset_spec("eurosat"), output_dir)


def download_resisc45(output_dir: Path) -> None:
    """Download torchgeo's NWPU-RESISC45 into ``output_dir/resisc45`` for all splits.

    All splits share one 427 MB archive; verify its checksum before loading samples.
    """
    _download_torchgeo(get_dataset_spec("resisc45"), output_dir)


def _download_torchgeo(spec: DatasetSpec, output_dir: Path) -> None:
    import torchgeo.datasets

    for name in list_datasets(source="torchgeo"):
        sibling = get_dataset_spec(name)
        source = sibling.source
        assert isinstance(source, TorchGeoSource)
        if sibling.storage_name != spec.storage_name:
            continue
        target = Path(output_dir) / Path(source.root).relative_to("data")
        target.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading torchgeo %s -> %s", source.upstream_class, target)
        cls = getattr(torchgeo.datasets, source.upstream_class)
        kwargs = {"checksum": True} if source.download_checksum else {}
        for split in sibling.split_sizes:
            cls(root=str(target), split=split, download=True, **kwargs)


def download_datasets(names: list[str], output_dir: Path = Path("data")) -> None:
    """Download individually named benchmark datasets.

    Args:
        names: Dataset names from :data:`DOWNLOADABLE_DATASETS`.
        output_dir: Benchmark data root.

    Raises:
        ValueError: If a name is unknown or *names* is empty.
    """
    selected = _validate_names(names)
    v1 = [name for name in selected if name in V1_DATASETS]
    v2 = [name for name in selected if name in DEFAULT_V2_DATASETS]
    for name in v1:
        download_geobench_v1(output_dir, datasets=[name])
    if v2:
        download_geobench_v2(output_dir, datasets=v2)
    storage = {
        get_dataset_spec(name).storage_name: get_dataset_spec(name)
        for name in selected
        if get_dataset_spec(name).source.kind == "torchgeo"
    }
    for spec in storage.values():
        _download_torchgeo(spec, output_dir)
