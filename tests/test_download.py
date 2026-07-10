"""Unit tests for dataset download helpers."""

import logging
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from torchgeo_bench.download import (
    DEFAULT_V2_DATASETS,
    EUROSAT_ARCHIVE_NAME,
    EUROSAT_BASE_DIR,
    EUROSAT_SPLIT_FILENAMES,
    _decompress_zip_with_progress,
    download_eurosat,
    download_geobench_v1,
    download_geobench_v2,
)


def test_decompress_zip_with_progress_removes_archive_and_logs_cleanup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("classification_v1.0/sample.txt", "payload")

    caplog.set_level(logging.INFO, logger="torchgeo_bench.download")
    _decompress_zip_with_progress(archive, tmp_path)

    assert not archive.exists()
    assert (tmp_path / "classification_v1.0" / "sample.txt").read_text() == "payload"
    assert any("archive.zip" in message and "reclaimed" in message for message in caplog.messages)


def test_download_geobench_v1_creates_output_and_decompresses(tmp_path: Path) -> None:
    out = tmp_path / "data"

    def _fake_snapshot_download(*, repo_id: str, repo_type: str, local_dir: Path) -> None:
        del repo_id, repo_type
        nested = local_dir / "classification_v1.0"
        nested.mkdir(parents=True, exist_ok=True)
        (nested / "archive.zip").write_bytes(b"placeholder")

    with (
        mock.patch(
            "torchgeo_bench.download.snapshot_download", side_effect=_fake_snapshot_download
        ),
        mock.patch("torchgeo_bench.download._decompress_zip_with_progress") as decompress_mock,
    ):
        download_geobench_v1(out)

    assert out.exists()
    decompress_mock.assert_called_once()


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


def test_download_eurosat_cleans_archive_after_success(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "data"
    target = out / "eurosat"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / EUROSAT_ARCHIVE_NAME
    archive.write_bytes(b"archive-bytes")
    called_splits: list[str] = []

    def _fake_eurosat(*, root: str, split: str, download: bool) -> None:
        assert download is True
        called_splits.append(split)
        dataset_root = Path(root)
        (dataset_root / EUROSAT_BASE_DIR).mkdir(parents=True, exist_ok=True)
        for split_filename in EUROSAT_SPLIT_FILENAMES:
            (dataset_root / split_filename).write_text(split_filename)

    caplog.set_level(logging.INFO, logger="torchgeo_bench.download")
    with mock.patch("torchgeo_bench.download.EuroSAT", side_effect=_fake_eurosat):
        download_eurosat(out)

    assert target.exists()
    assert called_splits == ["train", "val", "test"]
    assert not archive.exists()
    assert any(
        EUROSAT_ARCHIVE_NAME in message and "reclaimed" in message for message in caplog.messages
    )


def test_download_eurosat_keeps_archive_when_split_setup_fails(tmp_path: Path) -> None:
    out = tmp_path / "data"
    target = out / "eurosat"
    target.mkdir(parents=True, exist_ok=True)
    archive = target / EUROSAT_ARCHIVE_NAME
    archive.write_bytes(b"archive-bytes")

    def _fake_eurosat(*, root: str, split: str, download: bool) -> None:
        assert download is True
        dataset_root = Path(root)
        (dataset_root / EUROSAT_BASE_DIR).mkdir(parents=True, exist_ok=True)
        (dataset_root / f"partial-{split}.txt").write_text(split)
        if split == "val":
            raise RuntimeError("split setup failed")

    with (
        mock.patch("torchgeo_bench.download.EuroSAT", side_effect=_fake_eurosat),
        pytest.raises(RuntimeError, match="split setup failed"),
    ):
        download_eurosat(out)

    assert archive.exists()


def test_download_eurosat_is_idempotent_when_archive_is_missing(tmp_path: Path) -> None:
    out = tmp_path / "data"
    target = out / "eurosat"
    (target / EUROSAT_BASE_DIR).mkdir(parents=True, exist_ok=True)
    for split_filename in EUROSAT_SPLIT_FILENAMES:
        (target / split_filename).write_text(split_filename)

    with mock.patch("torchgeo_bench.download.EuroSAT") as eurosat_mock:
        download_eurosat(out)

    assert not (target / EUROSAT_ARCHIVE_NAME).exists()
    called_splits = [kwargs["split"] for _, kwargs in eurosat_mock.call_args_list]
    assert called_splits == ["train", "val", "test"]


def test_download_eurosat_cleans_stale_archive_when_already_extracted(tmp_path: Path) -> None:
    out = tmp_path / "data"
    target = out / "eurosat"
    (target / EUROSAT_BASE_DIR).mkdir(parents=True, exist_ok=True)
    for split_filename in EUROSAT_SPLIT_FILENAMES:
        (target / split_filename).write_text(split_filename)
    archive = target / EUROSAT_ARCHIVE_NAME
    archive.write_bytes(b"archive-bytes")

    with mock.patch("torchgeo_bench.download.EuroSAT"):
        download_eurosat(out)

    assert not archive.exists()
