"""Read GeoBench V1 from indexed WebDataset tar shards.

Shards hold ``<sid>.bands.npz`` and ``<sid>.meta.json`` pairs for roughly 1000 samples.

Index byte offsets once; each worker opens its own file descriptors for fork-safe reads.
Acquisition and archive verification belong to ``torchgeo_bench._v1_download``;
this reader only opens local files and never downloads or converts data.
"""

import io
import json
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from ._metadata import decode_metadata
from .transforms import integer_target


def _index_shards(paths: list[Path]) -> dict[str, dict[str, tuple[Path, int, int]]]:
    index: dict[str, dict[str, tuple[Path, int, int]]] = {}
    for path in paths:
        with tarfile.open(path, "r:") as archive:
            for member in archive:
                for ext in ("bands.npz", "meta.json"):
                    suffix = "." + ext
                    if member.name.endswith(suffix):
                        if not member.isfile():
                            raise ValueError(f"Not a sample file: {member.name} in {path}.")
                        # IDs can contain dots; remove the suffix, not everything after a dot.
                        sample_id = member.name.removesuffix(suffix)
                        index.setdefault(sample_id, {})[ext] = (
                            path,
                            member.offset_data,
                            member.size,
                        )
                        break
    return index


class GeoBenchv1Sharded(Dataset):
    """GeoBench V1 dataset reading from WebDataset tar shards."""

    def __init__(  # noqa: PLR0913 - public dataset constructor.
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
            raise FileNotFoundError(
                f"Sharded dataset dir not found: {self.dataset_dir}. "
                f"Run `torchgeo-bench download geobench_v1 --datasets {dataset_name}`."
            )

        partition_file = self.dataset_dir / f"{partition}_partition.json"
        with open(partition_file) as f:
            partition_data = json.load(f)
        if split not in partition_data:
            message = (
                f"Split '{split}' not found in partition. Available: {list(partition_data.keys())}"
            )
            if split in ("train", "valid", "test"):
                raise FileNotFoundError(message)
            raise ValueError(message)
        self.sample_ids: list[str] = partition_data[split]
        self.transform = transform

        shard_paths = sorted(self.dataset_dir.glob("shard_*.tar"))
        if not shard_paths:
            raise FileNotFoundError(
                f"No shard_*.tar in {self.dataset_dir}. "
                f"Run `torchgeo-bench download geobench_v1 --datasets {dataset_name}`."
            )
        self._index = _index_shards(shard_paths)
        for sample_id in self.sample_ids:
            parts = self._index.get(sample_id, {})
            for ext in ("meta.json", "bands.npz"):
                if ext not in parts:
                    raise ValueError(
                        f"Sample {sample_id!r} requires '.{ext}' in {self.dataset_dir}. "
                        "Legacy pickle shards are not supported. Replace this cache with "
                        f"'torchgeo-bench download geobench_v1 --datasets {dataset_name}'."
                    )

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
        return decode_metadata(self._read(parts["meta.json"]))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict:
        sid = self.sample_ids[index]
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
            # NPZ insertion order chooses the first matching acquisition; never sort dates.
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
            label_t = integer_target(label_arr.item(), name=f"{self.dataset_dir.name} label")

        sample: dict = {"image": image_t, "label": label_t, "sample_id": sid}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
