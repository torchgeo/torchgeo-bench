"""Download GeoBench datasets and torchgeo EuroSAT into ``data/``.

Three targets:

- ``geobench_v1`` — full GeoBench V1 classification suite from
  ``recursix/geo-bench-1.0``. Downloads to ``<output>/`` (the HF repo already
  contains a top-level ``classification_v1.0/`` directory).
- ``geobench_v2`` — selected GeoBench V2 datasets from ``aialliance/<name>``
  HF repos. Defaults to the benchmark-supported datasets; override with
  ``--datasets``. Each dataset goes to ``<output>/geobenchv2/<name>``.
- ``eurosat`` — torchgeo's EuroSAT downloader, into ``<output>/eurosat``.
"""

import logging
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download
from rich.progress import track
from torchgeo.datasets import EuroSAT

logger = logging.getLogger(__name__)

GEOBENCH_V1_REPO = "recursix/geo-bench-1.0"
GEOBENCH_V2_REPO_PREFIX = "aialliance"
EUROSAT_ARCHIVE_NAME = EuroSAT.filename
EUROSAT_BASE_DIR = Path(EuroSAT.base_dir)
EUROSAT_SPLITS = tuple(EuroSAT.splits)
EUROSAT_SPLIT_FILENAMES = tuple(EuroSAT.split_filenames[split] for split in EUROSAT_SPLITS)

# Default V2 datasets to download — only those the benchmark runner knows about.
# Sourced from torchgeo_bench.datasets.geobench_v2._V2_REGISTRY at module load.
DEFAULT_V2_DATASETS: tuple[str, ...] = (
    "benv2",
    "burn_scars",
    "caffe",
    "cloudsen12",
    "dynamic_earthnet",
    "flair2",
    "forestnet",
    "fotw",
    "kuro_siwo",
    "pastis",
    "so2sat",
    "spacenet2",
    "spacenet7",
    "treesatai",
)


def _format_bytes(num_bytes: int) -> str:
    """Return a human-readable file size."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def _remove_download_artifact(path: Path, *, reason: str) -> None:
    """Delete a download artifact and log reclaimed disk space."""
    try:
        size_bytes = path.stat().st_size
    except FileNotFoundError:
        return

    try:
        path.unlink()
    except FileNotFoundError:
        return

    logger.info("Removed %s after %s (%s reclaimed)", path, reason, _format_bytes(size_bytes))


def _decompress_zip_with_progress(zip_path: Path, extract_to: Path) -> None:
    """Extract ``zip_path`` into ``extract_to`` with a progress bar."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in track(zf.namelist(), description=f"Extracting {zip_path.name}"):
            zf.extract(name, extract_to)
    _remove_download_artifact(zip_path, reason=f"successfully extracting {zip_path.name}")


def _eurosat_is_ready(target: Path) -> bool:
    """Return whether EuroSAT extraction and split setup are complete."""
    image_dir = target / EUROSAT_BASE_DIR
    split_files = [target / split_filename for split_filename in EUROSAT_SPLIT_FILENAMES]
    return image_dir.is_dir() and all(split_file.is_file() for split_file in split_files)


def _cleanup_eurosat_archive(target: Path) -> None:
    """Remove the downloaded EuroSAT archive once extracted data is ready."""
    archive = target / EUROSAT_ARCHIVE_NAME
    if not archive.exists():
        return
    if not _eurosat_is_ready(target):
        logger.info("Keeping %s because EuroSAT extraction or split setup is incomplete.", archive)
        return
    _remove_download_artifact(
        archive,
        reason="successful EuroSAT extraction and split setup",
    )


def download_geobench_v1(output_dir: Path) -> None:
    """Download GeoBench V1 to ``output_dir`` (creates ``classification_v1.0/`` inside)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GeoBench v1 from %s -> %s", GEOBENCH_V1_REPO, output_dir)

    snapshot_download(
        repo_id=GEOBENCH_V1_REPO,
        repo_type="dataset",
        local_dir=output_dir,
    )

    for zip_path in sorted(output_dir.rglob("*.zip")):
        logger.info("Decompressing %s", zip_path)
        _decompress_zip_with_progress(zip_path, zip_path.parent)

    logger.info("GeoBench v1 download complete.")


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
    names = datasets or list(DEFAULT_V2_DATASETS)
    logger.info("Downloading %d GeoBench v2 dataset(s) to %s", len(names), v2_root)
    for name in names:
        download_geobench_v2_dataset(name, v2_root)
    logger.info("GeoBench v2 download complete.")


def download_eurosat(output_dir: Path) -> None:
    """Download torchgeo's EuroSAT into ``output_dir/eurosat`` for all splits."""
    target = Path(output_dir) / "eurosat"
    target.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading torchgeo EuroSAT -> %s", target)
    for split in EUROSAT_SPLITS:
        EuroSAT(root=str(target), split=split, download=True)
    _cleanup_eurosat_archive(target)
    logger.info("EuroSAT download complete.")
