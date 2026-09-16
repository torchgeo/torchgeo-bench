"""Small, disjoint on-disk datasets read by the production dataset adapters."""

import io
import json
import struct
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.transform import Affine

from torchgeo_bench.datasets import get_dataset_spec


def require_dataset_data(name: str) -> None:
    """Skip absent optional real data, never an incompatible existing cache."""
    spec = get_dataset_spec(name)
    source = spec.source
    path = Path(source.root)
    if source.kind != "torchgeo":
        path /= spec.storage_name
    if not path.exists():
        pytest.skip(f"{name} data not supplied; expected {path}")


def write_v1_sample(
    archive: tarfile.TarFile, sample_id: str, arrays: dict[str, np.ndarray], metadata: dict
) -> None:
    """Write the mirror's JSON/NPZ members without any intermediate representation."""
    pixels = io.BytesIO()
    np.savez(pixels, **arrays)
    for suffix, payload in (
        ("bands.npz", pixels.getvalue()),
        ("meta.json", json.dumps(metadata, allow_nan=False).encode()),
    ):
        member = tarfile.TarInfo(f"{sample_id}.{suffix}")
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


def write_classification_files(
    root: Path,
    dataset_name: str,
    labels: tuple[int | list[int], ...],
    *,
    all_bands: bool = False,
    splits: tuple[str, ...] = ("train", "valid", "test"),
) -> Path:
    """Write separable V1 JSON/NPZ shards with distinct IDs and pixels in each split."""
    bench = get_dataset_spec(dataset_name)
    directory = root / bench.source.root / bench.storage_name
    directory.mkdir(parents=True)
    specs = bench.select_band_specs(None if all_bands else tuple(bench.rgb_bands))
    partition: dict[str, list[str]] = {}
    for split_index, (split, per_class) in enumerate((("train", 12), ("valid", 4), ("test", 4))):
        if split not in splits:
            continue
        partition[split] = []
        with tarfile.open(directory / f"shard_{split_index:05d}.tar", "w") as archive:
            for class_index, label in enumerate(labels):
                for index in range(per_class):
                    sample_id = f"{split}-{class_index}-{index}"
                    partition[split].append(sample_id)
                    signal = 0.5 + 2.5 * class_index + 0.03 * split_index + 0.001 * index
                    arrays = {
                        band.source_name: np.full(
                            (16, 16), band.mean + band.std * signal, np.float32
                        )
                        for band in specs
                    }
                    write_v1_sample(
                        archive,
                        sample_id,
                        arrays,
                        {"label": label, "bands_order": [band.source_name for band in specs]},
                    )
    (directory / "default_partition.json").write_text(json.dumps(partition))
    return directory


def geotiff_bytes(pixels: np.ndarray) -> bytes:
    """Encode channel-first pixels as a real, georeferenced in-memory GeoTIFF."""
    with MemoryFile() as memory:
        with memory.open(
            driver="GTiff",
            height=pixels.shape[-2],
            width=pixels.shape[-1],
            count=pixels.shape[0],
            dtype=pixels.dtype,
            crs="EPSG:4326",
            transform=Affine(0.01, 0, 10, 0, -0.01, 45),
        ) as dataset:
            dataset.write(pixels)
        return memory.read()


def _tortilla_bytes(samples: list[tuple[dict[str, Any], bytes]]) -> bytes:
    # Tacoreader's 18-byte header stores little-endian Parquet footer offset and size.
    contents = bytearray(18)
    rows = []
    for metadata, payload in samples:
        rows.append({**metadata, "tortilla:offset": len(contents), "tortilla:length": len(payload)})
        contents.extend(payload)
    footer = pd.DataFrame(rows).to_parquet(index=False)
    contents[:18] = struct.pack("<2sQQ", b"#y", len(contents), len(footer))
    contents.extend(footer)
    return bytes(contents)


def write_caffe_files(root: Path, *, target_class: int | None = None) -> Path:
    """Write a real nested Taco containing disjoint grayscale/mask GeoTIFF samples."""
    directory = root / "data" / "geobenchv2" / "caffe"
    directory.mkdir(parents=True)
    samples = []
    for split_index, (split, count) in enumerate((("train", 6), ("validation", 4), ("test", 4))):
        for index in range(count):
            y, x = np.indices((16, 16))
            mask = ((x // 8 + 2 * (y // 8) + index) % 4).astype(np.uint8)
            if target_class is not None:
                mask.fill(target_class)
            pixels = 20 + mask.astype(np.float32) * 55 + split_index * 0.5 + index * 0.02
            mask[0, 0] = 255
            sample_id = f"{split}-{index}"
            assets = [
                (
                    {
                        "tortilla:id": f"{sample_id}-{kind}",
                        "tortilla:file_format": "GTiff",
                        "stac:centroid": "POINT (10 45)",
                    },
                    geotiff_bytes(array[None]),
                )
                for kind, array in (("image", pixels), ("mask", mask))
            ]
            samples.append(
                (
                    {
                        "tortilla:id": sample_id,
                        "tortilla:data_split": split,
                        "tortilla:file_format": "TORTILLA",
                    },
                    _tortilla_bytes(assets),
                )
            )
    (directory / "geobench_caffe.tortilla").write_bytes(_tortilla_bytes(samples))
    return directory


def write_torchgeo_download_files(root: Path, name: str) -> Path:
    """Create small imagery ZIPs and standard/spatial split files for HTTP downloads."""
    from torchgeo.datasets import RESISC45, EuroSAT, EuroSATSpatial

    dataset_class = EuroSAT if name == "eurosat" else RESISC45
    directory = root / name
    directory.mkdir(parents=True)
    image_root = EuroSAT.base_dir if name == "eurosat" else RESISC45.directory
    classes = ("AnnualCrop", "Forest") if name == "eurosat" else ("airplane", "forest")
    split_ids: dict[str, list[str]] = {split: [] for split in ("train", "val", "test")}
    with zipfile.ZipFile(directory / dataset_class.filename, "w") as archive:
        for split_index, (split, ids) in enumerate(split_ids.items()):
            for class_index, label in enumerate(classes):
                suffix = "tif" if name == "eurosat" else "jpg"
                sample_id = f"{label}_{split}.{suffix}"
                ids.append(sample_id)
                value = 20 + class_index * 100 + split_index
                if name == "eurosat":
                    payload = geotiff_bytes(np.full((13, 16, 16), value, np.uint16))
                else:
                    buffer = io.BytesIO()
                    Image.fromarray(np.full((16, 16, 3), value, np.uint8)).save(buffer, "JPEG")
                    payload = buffer.getvalue()
                archive.writestr(f"{image_root}/{label}/{sample_id}", payload)
    for split, ids in split_ids.items():
        filenames = (
            [EuroSAT.split_filenames[split], EuroSATSpatial.split_filenames[split]]
            if name == "eurosat"
            else [f"resisc45-{split}.txt"]
        )
        for filename in filenames:
            (directory / filename).write_text("\n".join(ids) + "\n")
    return directory
