"""Download GeoBench datasets and torchgeo EuroSAT into ``data/``.

Targets:

- ``geobench_v1`` — pickle-free GeoBench V1 classification shards from
  ``calebrob6/geobenchv1-webdataset`` under ``<output>/classification_v1.0_wds/``.
- ``geobench_v2`` — selected GeoBench V2 datasets from ``aialliance/<name>``
  HF repos. Defaults to the benchmark-supported datasets; override with
  ``--datasets``. Each dataset goes to ``<output>/geobenchv2/<name>``.
- ``eurosat`` — torchgeo's EuroSAT downloader, into ``<output>/eurosat``.
- ``resisc45`` — torchgeo's NWPU-RESISC45 downloader, into ``<output>/resisc45``.
"""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download
from torchgeo.datasets import RESISC45, EuroSAT, EuroSATSpatial

from torchgeo_bench.datasets._v1_webdataset import download_sharded_root
from torchgeo_bench.datasets.geobench_v2 import list_v2_datasets

logger = logging.getLogger(__name__)

GEOBENCH_V2_REPO_PREFIX = "aialliance"

DEFAULT_V2_DATASETS: tuple[str, ...] = tuple(list_v2_datasets())


def download_geobench_v1(output_dir: Path, datasets: list[str] | None = None) -> None:
    """Download verified, pickle-free GeoBench V1 shards to ``output_dir``.

    Args:
        output_dir: Benchmark data root (typically ``data/``).
        datasets: Specific dataset names to fetch from the sharded mirror.
            ``None`` downloads all six classification datasets.

    Raises:
        ValueError: If dataset names are invalid or an archive checksum fails.
    """
    download_sharded_root(Path(output_dir) / "classification_v1.0_wds", datasets)


def download_geobench_v2_dataset(name: str, v2_root: Path) -> None:
    """Download a single GeoBench V2 dataset into ``v2_root/<name>``."""
    target = v2_root / name
    target.mkdir(parents=True, exist_ok=True)
    repo_id = f"{GEOBENCH_V2_REPO_PREFIX}/{name}"
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
    v2_root = Path(output_dir) / "geobenchv2"
    v2_root.mkdir(parents=True, exist_ok=True)
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
    logger.info("Downloading %d GeoBench v2 dataset(s) to %s", len(names), v2_root)
    for name in names:
        download_geobench_v2_dataset(name, v2_root)
    logger.info("GeoBench v2 download complete.")


def download_eurosat(output_dir: Path) -> None:
    """Download EuroSAT imagery and both standard/spatial split definitions."""
    target = Path(output_dir) / "eurosat"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading torchgeo EuroSAT -> %s", target)
    for dataset_cls in (EuroSAT, EuroSATSpatial):
        for split in ("train", "val", "test"):
            dataset_cls(root=str(target), split=split, download=True)
    logger.info("EuroSAT download complete.")


def download_resisc45(output_dir: Path) -> None:
    """Download torchgeo's NWPU-RESISC45 into ``output_dir/resisc45`` for all splits.

    The three splits share one 427MB archive, so only the first call fetches
    it; the rest just read their split file.  ``checksum=True`` because the
    archive is served from a pinned Hugging Face revision and a truncated
    download would otherwise surface as missing classes much later.
    """
    target = Path(output_dir) / "resisc45"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading torchgeo RESISC45 -> %s", target)
    for split in ("train", "val", "test"):
        RESISC45(root=str(target), split=split, download=True, checksum=True)
    logger.info("RESISC45 download complete.")
