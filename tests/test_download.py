"""Unit tests for dataset download helpers."""

from pathlib import Path
from unittest import mock

from torchgeo_bench.download import (
    DEFAULT_V2_DATASETS,
    download_eurosat,
    download_geobench_v1,
    download_geobench_v2,
)


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


def test_download_eurosat_creates_target_and_downloads_splits(tmp_path: Path) -> None:
    out = tmp_path / "data"
    with mock.patch("torchgeo_bench.download.EuroSAT") as eurosat_mock:
        download_eurosat(out)

    assert (out / "eurosat").exists()
    called_splits = [kwargs["split"] for _, kwargs in eurosat_mock.call_args_list]
    assert called_splits == ["train", "val", "test"]
