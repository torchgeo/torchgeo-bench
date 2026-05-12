"""WebDataset-backed loader for the GeoBench V1 sharded layout.

Drops the per-sample HDF5 file-open from ``__getitem__`` (one NFS round-trip
per sample) by reading from ~22 tar shards instead.  Format is produced by
``scripts/repack_geobench_v1.py`` and mirrored on the Hub at
``isaaccorley/geobenchv1-webdataset`` (auto-pulled by
:func:`ensure_sharded_root` when no local copy is present).

Each shard contains ``<sid>.bands.npz`` and ``<sid>.meta.pkl`` files for
~1000 samples.  Indexing happens once in ``__init__``: every sample's byte
range inside its shard is recorded as ``(shard_path, offset, size)`` so
``__getitem__`` does a plain ``open()`` + ``seek()`` + ``read()`` and
avoids the ``tarfile`` state machine entirely.  This is fork-safe (each
worker opens its own file descriptors) and faster (no per-call tar header
parsing).

Output dict matches :class:`~torchgeo_bench.datasets.geobench_v1.GeoBenchv1`
exactly.
"""

import io
import json
import logging
import os
import pickle
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

V1_HF_REPO_ID = "isaaccorley/geobenchv1-webdataset"


def ensure_sharded_root(
    dataset_name: str,
    sharded_root: Path,
    *,
    repo_id: str = V1_HF_REPO_ID,
    cache_dir: str | os.PathLike | None = None,
) -> Path:
    """Snapshot-download the sharded V1 mirror into ``sharded_root`` if absent.

    Pulls only the requested ``dataset_name`` subdirectory so a single-dataset
    run doesn't download the full 35 GB collection.  Returns the local path of
    the dataset directory (whether downloaded or already present).
    """
    target = Path(sharded_root) / dataset_name
    if target.exists() and any(target.glob("shard_*.tar")):
        return target

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required to auto-download the GeoBench V1 "
            "WebDataset mirror.  Install via `pip install huggingface_hub`, "
            "or set GEOBENCH_V1_NO_HF_DOWNLOAD=1 and provide the data manually."
        ) from e

    sharded_root = Path(sharded_root)
    sharded_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s/%s -> %s", repo_id, dataset_name, sharded_root)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=sharded_root,
        allow_patterns=[f"{dataset_name}/*"],
        cache_dir=cache_dir,
    )
    if not any(target.glob("shard_*.tar")):
        raise RuntimeError(
            f"Auto-download of {repo_id}/{dataset_name} produced no shards under "
            f"{target}; check the repo layout."
        )
    return target


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> type:  # type: ignore[override]
        if module == "geobench.dataset":
            return type(name, (), {})
        return super().find_class(module, name)


def _safe_unpickle(b: bytes) -> dict:
    try:
        return pickle.loads(b)
    except (ModuleNotFoundError, AttributeError):
        return _StubUnpickler(io.BytesIO(b)).load()


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

        # Index every member: sid -> {"bands.npz": (path, offset, size), "meta.pkl": ...}
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
                    for ext in ("bands.npz", "meta.pkl"):
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
        return _safe_unpickle(self._read(self._index[sample_id]["meta.pkl"]))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        parts = self._index[sid]
        bands_dict = dict(np.load(io.BytesIO(self._read(parts["bands.npz"]))))
        meta = _safe_unpickle(self._read(parts["meta.pkl"]))

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
