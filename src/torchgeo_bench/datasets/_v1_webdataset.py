"""WebDataset-backed loader for the GeoBench V1 sharded layout.

Drops the per-sample HDF5 file-open from ``__getitem__`` (one NFS round-trip
per sample) by reading from ~22 tar shards instead. The data-only format is
produced by ``experiments/scripts/repack_geobench_v1.py``.
:func:`download_sharded_root` fetches the pinned, pickle-free Hub mirror and
verifies each downloaded tar archive against the bundled SHA-256 list.

Each shard contains ``<sid>.bands.npz`` and ``<sid>.meta.json`` files for
~1000 samples.  Indexing happens once in ``__init__``: every sample's byte
range inside its shard is recorded as ``(shard_path, offset, size)`` so
``__getitem__`` does a plain ``open()`` + ``seek()`` + ``read()`` and
avoids the ``tarfile`` state machine entirely.  This is fork-safe (each
worker opens its own file descriptors) and faster (no per-call tar header
parsing).

Output dict matches :class:`~torchgeo_bench.datasets.geobench_v1.GeoBenchv1`
exactly.
"""

import hashlib
import io
import json
import logging
import os
import tarfile
from collections.abc import Callable
from functools import cache
from importlib.resources import files
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from ._metadata import decode_metadata

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
    cache_dir: str | os.PathLike | None = None,
) -> None:
    """Download and verify the selected pickle-free V1 datasets.

    Args:
        sharded_root: Destination collection directory.
        datasets: Dataset names, or ``None`` for the full classification suite.
        cache_dir: Optional Hugging Face download cache.
    """
    checksums = _shard_checksums()
    available = {name.split("/", 1)[0] for name in checksums}
    names = sorted(available) if datasets is None else list(dict.fromkeys(datasets))
    if not names:
        raise ValueError("datasets must contain at least one GeoBench V1 dataset name")
    unknown = sorted(set(names) - available)
    if unknown:
        raise ValueError(f"Unknown GeoBench V1 dataset(s): {', '.join(unknown)}.")
    sharded_root = Path(sharded_root)
    sharded_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GeoBench V1 from %s -> %s", V1_HF_REPO_ID, sharded_root)
    snapshot_download(
        repo_id=V1_HF_REPO_ID,
        repo_type="dataset",
        revision=V1_HF_REVISION,
        local_dir=sharded_root,
        allow_patterns=[f"{name}/*" for name in names],
        cache_dir=cache_dir,
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


class GeoBenchv1Sharded(Dataset):
    """GeoBench V1 dataset reading from WebDataset tar shards."""

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: Literal["train", "valid", "test"],
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable[[dict], dict] | None = None,
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(root) / dataset_name
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Sharded dataset dir not found: {self.dataset_dir}")

        partition_file = self.dataset_dir / f"{partition}_partition.json"
        with open(partition_file) as f:
            partition_data = json.load(f)
        if split not in partition_data:
            raise ValueError(
                f"Split '{split}' not found in partition. Available: {list(partition_data.keys())}"
            )
        self.sample_ids: list[str] = partition_data[split]
        self.transform = transform

        # Index every member: sid -> {suffix: (path, offset, size)}.
        shard_paths = sorted(self.dataset_dir.glob("shard_*.tar"))
        if not shard_paths:
            raise FileNotFoundError(f"No shard_*.tar in {self.dataset_dir}")
        # Sample IDs may contain dots (m-forestnet uses
        # ``<lat>_<lon>_<date>.hdf5``), so split on the known suffix instead
        # of the first ``.``.
        self._index: dict[str, dict[str, tuple[Path, int, int]]] = {}
        for path in shard_paths:
            with tarfile.open(path, "r") as t:
                for m in t.getmembers():
                    for ext in ("bands.npz", "meta.json"):
                        suffix = "." + ext
                        if m.name.endswith(suffix):
                            base = m.name[: -len(suffix)]
                            self._index.setdefault(base, {})[ext] = (
                                path,
                                m.offset_data,
                                m.size,
                            )
                            break

        if bands is None:
            sample_meta = self._load_meta(self.sample_ids[0])
            self.band_names: list[str] = list(sample_meta["bands_order"])
        else:
            self.band_names = list(bands)

    def _read(self, ref: tuple[Path, int, int]) -> bytes:
        path, offset, size = ref
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read(size)

    def _load_meta(self, sample_id: str) -> dict:
        parts = self._index[sample_id]
        if "meta.json" not in parts:
            raise ValueError(
                f"Sample {sample_id!r} requires '.meta.json' metadata. "
                "Legacy pickle shards are not supported. Replace this cache with "
                f"'torchgeo-bench download geobench_v1 --datasets {self.dataset_dir.name}'."
            )
        return decode_metadata(self._read(parts["meta.json"]))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        parts = self._index[sid]
        meta = self._load_meta(sid)
        with np.load(io.BytesIO(self._read(parts["bands.npz"])), allow_pickle=False) as archive:
            bands_dict = {name: archive[name] for name in archive.files}

        bands_data = []
        available = list(bands_dict)
        for band_name in self.band_names:
            if band_name in bands_dict:
                bands_data.append(bands_dict[band_name])
                continue
            matching = [k for k in available if k.startswith(band_name)]
            if not matching:
                raise KeyError(
                    f"Band '{band_name}' not found in shard sample {sid}. "
                    f"Available: {available[:5]}..."
                )
            bands_data.append(bands_dict[matching[0]])

        image = np.stack(bands_data, axis=0).astype(np.float32)
        image_t = torch.from_numpy(image)

        label = meta["label"]
        label_arr = np.asarray(label)
        if label_arr.ndim > 0:
            label_t: torch.Tensor = torch.from_numpy(label_arr.astype(np.float32))
        else:
            label_t = torch.tensor(label_arr.item(), dtype=torch.long)

        sample: dict = {"image": image_t, "label": label_t, "sample_id": sid}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
