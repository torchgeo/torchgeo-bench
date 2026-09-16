"""Explicit acquisition and archive verification for the pinned V1 JSON mirror."""

import hashlib
import logging
import os
from functools import cache
from importlib.resources import files
from pathlib import Path

from huggingface_hub import snapshot_download

from .datasets import get_dataset_spec, list_datasets

logger = logging.getLogger(__name__)

V1_HF_REPO_ID = "calebrob6/geobenchv1-webdataset"
V1_HF_REVISION = "18c293d3a963c73e8e055a2fef6fca9e029c6e95"


@cache
def _shard_checksums() -> dict[str, str]:
    with files("torchgeo_bench.datasets").joinpath("_v1_checksums.sha256").open("r") as stream:
        return {name: checksum for checksum, name in (line.split() for line in stream)}


def download_sharded_root(
    sharded_root: Path,
    datasets: list[str] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
) -> None:
    """Download and verify the selected pickle-free V1 datasets.

    Args:
        sharded_root: Destination collection directory.
        datasets: Storage names, or ``None`` for the full classification suite.
        cache_dir: Optional Hugging Face download cache.
    """
    available = {get_dataset_spec(name).storage_name for name in list_datasets(source="v1")}
    names = sorted(available) if datasets is None else list(dict.fromkeys(datasets))
    if not names:
        raise ValueError("datasets must contain at least one GeoBench V1 dataset name")
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"Unknown GeoBench V1 dataset(s): {', '.join(unknown)}.")
    checksums = _shard_checksums()
    unchecked = sorted(set(names) - {path.split("/", 1)[0] for path in checksums})
    if unchecked:
        raise ValueError(
            f"No archive checksums for GeoBench V1 dataset(s): {', '.join(unchecked)}."
        )
    sharded_root = Path(sharded_root)
    sharded_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GeoBench V1 from %s -> %s", V1_HF_REPO_ID, sharded_root)
    snapshot_download(
        repo_id=V1_HF_REPO_ID,
        repo_type="dataset",
        revision=V1_HF_REVISION,
        local_dir=sharded_root,
        allow_patterns=[f"{name}/*" for name in names],
        cache_dir=Path(cache_dir) if cache_dir is not None else None,
    )
    logger.info("Verifying GeoBench V1 archive checksums.")
    for name, expected in checksums.items():
        if name.split("/", 1)[0] not in names:
            continue
        path = sharded_root / name
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(
                f"GeoBench V1 archive checksum mismatch: {path}. "
                "Remove this file and retry the download."
            )
    logger.info("GeoBench V1 download complete.")
