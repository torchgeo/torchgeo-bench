"""Unit tests for dataset download helpers."""

import zipfile
from pathlib import Path
from unittest import mock

import pytest

from torchgeo_bench.datasets.geobench_v2 import list_v2_datasets
from torchgeo_bench.download import (
    DEFAULT_V2_DATASETS,
    download_eurosat,
    download_geobench_v1,
    download_geobench_v2,
    download_resisc45,
)


def test_download_geobench_v1_creates_output_and_decompresses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "data"

    def _fake_snapshot_download(*, repo_id: str, repo_type: str, local_dir: Path) -> None:
        del repo_id, repo_type
        nested = local_dir / "classification_v1.0"
        nested.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(nested / "archive.zip", "w") as archive:
            archive.writestr("dataset/samples.txt", "sample 1\n")
            archive.writestr("dataset/metadata.json", '{"num_classes": 10}')

    with mock.patch(
        "torchgeo_bench.download.snapshot_download", side_effect=_fake_snapshot_download
    ):
        download_geobench_v1(out)

    nested = out / "classification_v1.0"
    assert (nested / "dataset/samples.txt").read_text() == "sample 1\n"
    assert (nested / "dataset/metadata.json").read_text() == '{"num_classes": 10}'
    assert not (nested / "archive.zip").exists()
    assert "Extracting archive.zip" in capsys.readouterr().err


def test_download_geobench_v1_subset_uses_sharded_mirror(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("torchgeo_bench.download.snapshot_download") as download_mock:
        download_geobench_v1(out, datasets=["m-eurosat", "m-forestnet"])

    download_mock.assert_called_once_with(
        repo_id="isaaccorley/geobenchv1-webdataset",
        repo_type="dataset",
        local_dir=out / "classification_v1.0_wds",
        allow_patterns=["m-eurosat/*", "m-forestnet/*"],
    )


def test_download_geobench_v1_rejects_empty_subset(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one GeoBench V1"):
        download_geobench_v1(tmp_path, datasets=[])


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
