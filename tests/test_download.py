"""Unit tests for dataset download helpers."""

import hashlib
import io
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from torchgeo_bench.datasets import _v1_webdataset as v1
from torchgeo_bench.datasets import geobench_v1, get_bench_dataset_class, list_datasets
from torchgeo_bench.datasets.geobench_v1 import _V1Dataset
from torchgeo_bench.datasets.geobench_v2 import list_v2_datasets
from torchgeo_bench.download import (
    DEFAULT_V2_DATASETS,
    download_eurosat,
    download_geobench_v1,
    download_geobench_v2,
    download_resisc45,
)


@pytest.fixture
def v1_download(monkeypatch) -> Iterator[mock.MagicMock]:
    bands = ("04 - Red", "03 - Green", "02 - Blue")
    arrays = io.BytesIO()
    np.savez(arrays, **{name: np.ones((2, 2), dtype=np.float32) for name in bands})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, value in (
            ("sample.bands.npz", arrays.getvalue()),
            ("sample.meta.json", json.dumps({"label": 0, "bands_order": bands}).encode()),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    payload = buffer.getvalue()
    checksums = {
        f"{name}/shard_00000.tar": hashlib.sha256(payload).hexdigest()
        for name in ("m-eurosat", "m-forestnet")
    }
    monkeypatch.setattr(v1, "_shard_checksums", lambda: checksums)

    def download(*, local_dir: Path, allow_patterns: list[str], **kwargs) -> None:
        for pattern in allow_patterns:
            directory = local_dir / pattern.removesuffix("/*")
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / "shard_00000.tar"
            if not path.exists():
                path.write_bytes(payload)
            (directory / "default_partition.json").write_text(
                json.dumps(dict.fromkeys(("train", "valid", "test"), ["sample"]))
            )

    with mock.patch.object(v1, "snapshot_download", side_effect=download) as download_mock:
        yield download_mock


@pytest.mark.parametrize("names", [None, ["m-eurosat"], ["m-forestnet", "m-eurosat", "m-eurosat"]])
def test_download_geobench_v1_uses_verified_json_shards(
    tmp_path: Path, v1_download: mock.MagicMock, names: list[str] | None
) -> None:
    download_geobench_v1(tmp_path, datasets=names)
    selected = ["m-eurosat", "m-forestnet"] if names is None else list(dict.fromkeys(names))
    v1_download.assert_called_once_with(
        repo_id="calebrob6/geobenchv1-webdataset",
        repo_type="dataset",
        revision="18c293d3a963c73e8e055a2fef6fca9e029c6e95",
        local_dir=tmp_path / "classification_v1.0_wds",
        allow_patterns=[f"{name}/*" for name in selected],
        cache_dir=None,
    )
    assert not (tmp_path / "classification_v1.0").exists()
    for name in selected:
        sample = v1.GeoBenchv1Sharded(tmp_path / "classification_v1.0_wds", name, "train")[0]
        assert sample["image"].shape == (3, 2, 2)
        assert sample["label"].item() == 0


def test_v1_download_rejects_corrupt_cached_archives(tmp_path: Path, v1_download) -> None:
    download_geobench_v1(tmp_path, datasets=["m-eurosat"])
    archive = tmp_path / "classification_v1.0_wds/m-eurosat/shard_00000.tar"
    archive.write_bytes(b"corrupt cached download")
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        download_geobench_v1(tmp_path, datasets=["m-eurosat"])


def test_v1_download_requires_every_expected_archive(tmp_path: Path, v1_download) -> None:
    v1_download.side_effect = None
    with pytest.raises(FileNotFoundError, match="shard_00000.tar"):
        download_geobench_v1(tmp_path, datasets=["m-eurosat"])


@pytest.mark.parametrize("names", [[], ["unknown"], ["../m-eurosat"]])
def test_v1_download_rejects_invalid_names(tmp_path: Path, v1_download, names) -> None:
    with pytest.raises(ValueError):
        download_geobench_v1(tmp_path, datasets=names)
    v1_download.assert_not_called()


def test_v1_auto_download_uses_the_shared_verified_reader(
    tmp_path: Path, v1_download, monkeypatch
) -> None:
    root = tmp_path / "shards"
    monkeypatch.setattr(geobench_v1, "V1_ROOT", tmp_path / "hdf5")
    monkeypatch.setattr(geobench_v1, "V1_SHARDED_ROOT", root)
    monkeypatch.delenv("GEOBENCH_V1_NO_HF_DOWNLOAD", raising=False)
    bench = get_bench_dataset_class("m-eurosat")()
    for split in ("train", "val", "test"):
        dataset = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
        assert dataset[0]["image"].shape == (3, 2, 2)
    v1_download.assert_called_once()
    assert v1_download.call_args.kwargs["local_dir"] == root
    assert v1_download.call_args.kwargs["allow_patterns"] == ["m-eurosat/*"]


def test_v1_archive_checksums_cover_the_published_suite() -> None:
    checksums = v1._shard_checksums()
    datasets = {
        name for name in list_datasets() if issubclass(get_bench_dataset_class(name), _V1Dataset)
    }
    assert len(checksums) == 89
    assert {name.split("/", 1)[0] for name in checksums} == datasets
    assert all(name.endswith(".tar") and len(value) == 64 for name, value in checksums.items())


def test_download_geobench_v2_subset(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("torchgeo_bench.download.download_geobench_v2_dataset") as dl_mock:
        download_geobench_v2(out, datasets=["burn_scars"])

    assert (out / "geobenchv2").exists()
    dl_mock.assert_called_once_with("burn_scars", out / "geobenchv2")


def test_download_geobench_v2_defaults_to_registry_list(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("torchgeo_bench.download.download_geobench_v2_dataset") as dl_mock:
        download_geobench_v2(out, datasets=None)

    assert dl_mock.call_count == len(DEFAULT_V2_DATASETS)
    assert tuple(list_v2_datasets()) == DEFAULT_V2_DATASETS


def test_download_geobench_v2_rejects_empty_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        download_geobench_v2(tmp_path, datasets=[])


def test_download_geobench_v2_rejects_unknown_dataset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown GeoBench V2"):
        download_geobench_v2(tmp_path, datasets=["not-a-dataset"])


def test_download_geobench_v2_deduplicates_names(tmp_path: Path) -> None:
    with mock.patch("torchgeo_bench.download.download_geobench_v2_dataset") as dl_mock:
        download_geobench_v2(tmp_path, datasets=["burn_scars", "burn_scars"])

    dl_mock.assert_called_once_with("burn_scars", tmp_path / "geobenchv2")


def test_download_eurosat_creates_target_and_downloads_splits(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with (
        mock.patch("torchgeo_bench.download.EuroSAT") as eurosat_mock,
        mock.patch("torchgeo_bench.download.EuroSATSpatial") as spatial_mock,
    ):
        download_eurosat(out)

    assert (out / "eurosat").exists()
    for dataset in (eurosat_mock, spatial_mock):
        assert dataset.call_args_list == [
            mock.call(root=str(out / "eurosat"), split=split, download=True)
            for split in ("train", "val", "test")
        ]


def test_download_resisc45_creates_target_and_downloads_splits(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("torchgeo_bench.download.RESISC45") as resisc_mock:
        download_resisc45(out)

    assert (out / "resisc45").exists()
    called_splits = [kwargs["split"] for _, kwargs in resisc_mock.call_args_list]
    assert called_splits == ["train", "val", "test"]


def test_download_resisc45_verifies_the_archive_checksum(tmp_path: Path) -> None:
    """The archive comes from a pinned HF revision; a truncated zip must fail loudly."""
    with mock.patch("torchgeo_bench.download.RESISC45") as resisc_mock:
        download_resisc45(tmp_path)

    assert all(kwargs["checksum"] for _, kwargs in resisc_mock.call_args_list)
