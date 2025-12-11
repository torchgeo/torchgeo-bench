"""Download GeoBench datasets from Hugging Face."""

import logging
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

logger = logging.getLogger(__name__)

# GeoBench v1 configuration
GEOBENCH_V1_REPO = "recursix/geo-bench-1.0"

# GeoBench v2 configuration - mapping dataset names to HuggingFace repos
GEOBENCH_V2_DATASETS = {
    "benv2": "aialliance/benv2",
    "biomassters": "aialliance/biomassters",
    "burn_scars": "aialliance/burn_scars",
    "caffe": "aialliance/caffe",
    "cloudsen12": "aialliance/cloudsen12",
    "dynamic_earthnet": "aialliance/dynamic_earthnet",
    "everwatch": "aialliance/everwatch",
    "flair2": "aialliance/flair2",
    "fotw": "aialliance/fotw",
    "kuro_siwo": "aialliance/kuro_siwo",
    "pastis": "aialliance/pastis",
    "spacenet2": "aialliance/spacenet2",
    "spacenet7": "aialliance/spacenet7",
    "substation": "aialliance/substation",
    "treesatai": "aialliance/treesatai",
    "wind_turbine": "aialliance/wind_turbine",
    "so2sat": "aialliance/so2sat",
    "forestnet": "aialliance/forestnet",
}

# GeoBench v2 file patterns - dataset name -> list of file patterns
GEOBENCH_V2_FILES = {
    "benv2": [
        "geobench_benv2.tortilla",
        "benv2_stats_satmae.json",
        "benv2_stats_clip_rescale.json",
        "README.md",
    ],
    "biomassters": [
        "geobench_biomassters.0000.part.tortilla",
        "geobench_biomassters.0001.part.tortilla",
        "geobench_biomassters.0002.part.tortilla",
        "biomassters_stats_satmae.json",
        "biomassters_stats_clip_rescale.json",
        "README.md",
    ],
    "burn_scars": [
        "geobench_burn_scars.tortilla",
        "burn_scars_stats_satmae.json",
        "burn_scars_stats_clip_rescale.json",
        "README.md",
    ],
    "caffe": [
        "geobench_caffe.tortilla",
        "caffe_stats_satmae.json",
        "caffe_stats_clip_rescale.json",
        "README.md",
    ],
    "cloudsen12": [
        "geobench_cloudsen12.tortilla",
        "cloudsen12_stats_satmae.json",
        "cloudsen12_stats_clip_rescale.json",
        "README.md",
    ],
    "dynamic_earthnet": [
        "geobench_dynamic_earthnet.0000.part.tortilla",
        "geobench_dynamic_earthnet.0001.part.tortilla",
        "geobench_dynamic_earthnet.0002.part.tortilla",
        "dynamic_earthnet_stats_satmae.json",
        "dynamic_earthnet_stats_clip_rescale.json",
        "README.md",
    ],
    "everwatch": [
        "geobench_everwatch.tortilla",
        "everwatch_stats_satmae.json",
        "everwatch_stats_clip_rescale.json",
        "README.md",
    ],
    "flair2": [
        "geobench_flair2.tortilla",
        "flair2_stats_satmae.json",
        "flair2_stats_clip_rescale.json",
        "README.md",
    ],
    "fotw": [
        "geobench_fotw.tortilla",
        "fotw_stats_satmae.json",
        "fotw_stats_clip_rescale.json",
        "README.md",
    ],
    "kuro_siwo": [
        "geobench_kuro_siwo.tortilla",
        "kuro_siwo_stats_satmae.json",
        "kuro_siwo_stats_clip_rescale.json",
        "README.md",
    ],
    "pastis": [
        "geobench_pastis.0000.part.tortilla",
        "geobench_pastis.0001.part.tortilla",
        "geobench_pastis.0002.part.tortilla",
        "pastis_stats_satmae.json",
        "pastis_stats_clip_rescale.json",
        "README.md",
    ],
    "spacenet2": [
        "geobench_spacenet2.tortilla",
        "spacenet2_stats_satmae.json",
        "spacenet2_stats_clip_rescale.json",
        "README.md",
    ],
    "spacenet7": [
        "geobench_spacenet7.tortilla",
        "spacenet7_stats_satmae.json",
        "spacenet7_stats_clip_rescale.json",
        "README.md",
    ],
    "substation": [
        "geobench_substation.tortilla",
        "substation_stats_satmae.json",
        "substation_stats_clip_rescale.json",
        "README.md",
    ],
    "treesatai": [
        "geobench_treesatai.tortilla",
        "treesatai_stats_satmae.json",
        "treesatai_stats_clip_rescale.json",
        "README.md",
    ],
    "wind_turbine": [
        "geobench_wind_turbine.tortilla",
        "wind_turbine_stats_satmae.json",
        "wind_turbine_stats_clip_rescale.json",
        "README.md",
    ],
    "so2sat": [
        "geobench_so2sat.tortilla",
        "so2sat_stats_satmae.json",
        "so2sat_stats_clip_rescale.json",
        "README.md",
    ],
    "forestnet": [
        "geobench_forestnet.tortilla",
        "forestnet_stats_satmae.json",
        "forestnet_stats_clip_rescale.json",
        "README.md",
    ],
}


def decompress_zip_with_progress(
    zip_file_path: Path, extract_to_folder: Path | None = None
) -> None:
    """Decompress a zip file with a progress bar and remove the zip file.

    Args:
        zip_file_path: Path to the zip file to decompress.
        extract_to_folder: Directory to extract files to. Defaults to zip file's parent directory.
    """
    extract_to_folder = extract_to_folder or zip_file_path.parent

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        file_names = zip_ref.namelist()

        with tqdm(
            total=len(file_names), unit="file", desc=f"Extracting {zip_file_path.name}"
        ) as pbar:
            for file in file_names:
                zip_ref.extract(file, extract_to_folder)
                pbar.update(1)

    zip_file_path.unlink()
    logger.info(f"Removed zip file: {zip_file_path}")


def download_geobench_v1(local_directory: Path | str, force: bool = False) -> None:
    """Download and extract the GeoBench v1 dataset from Hugging Face.

    Args:
        local_directory: Directory to download the dataset to.
        force: Force re-download of files even if they already exist.
    """
    local_directory = Path(local_directory)

    local_directory.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading GeoBench v1 dataset to: {local_directory}")

    api = HfApi()
    dataset_files = api.list_repo_files(repo_id=GEOBENCH_V1_REPO, repo_type="dataset")

    _download_files_v1(dataset_files, local_directory, force)
    _decompress_files(dataset_files, local_directory, force)

    logger.info("GeoBench v1 download and decompression completed.")


def _download_files_v1(
    dataset_files: list[str], local_directory: Path, force: bool = False
) -> None:
    """Download all files from the GeoBench v1 repository.

    Args:
        dataset_files: List of files to download.
        local_directory: Directory to download files to.
        force: Force re-download of files even if they already exist.
    """
    for file in dataset_files:
        local_file_path = local_directory / file

        # Skip if file already exists (unless force is enabled)
        if local_file_path.exists() and not force:
            logger.info(f"Skipping {file} (already exists)")
            continue

        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading {file}...")
        hf_hub_download(
            repo_id=GEOBENCH_V1_REPO,
            filename=file,
            cache_dir=local_directory,
            local_dir=local_directory,
            repo_type="dataset",
        )


def _decompress_files(
    dataset_files: list[str], local_directory: Path, force: bool = False
) -> None:
    """Decompress all zip files from the dataset.

    Args:
        dataset_files: List of all dataset files.
        local_directory: Directory containing zip files.
        force: Force re-extraction even if already extracted.
    """
    zip_files = [file for file in dataset_files if file.endswith(".zip")]

    for i, zip_file in enumerate(zip_files, start=1):
        zip_file_path = local_directory / zip_file

        # Skip if zip file doesn't exist (already extracted and removed) and not forcing
        if not zip_file_path.exists() and not force:
            logger.info(f"Skipping {zip_file} (already extracted)")
            continue

        logger.info(f"Decompressing {i}/{len(zip_files)}: {zip_file}...")
        decompress_zip_with_progress(zip_file_path)


def download_geobench_v2_dataset(
    dataset_name: str, local_directory: Path | str, force: bool = False
) -> None:
    """Download a single GeoBench v2 dataset from Hugging Face.

    Args:
        dataset_name: Name of the dataset to download.
        local_directory: Root directory to download datasets to.
        force: Force re-download of files even if they already exist.

    Raises:
        ValueError: If dataset_name is not recognized.
    """
    if dataset_name not in GEOBENCH_V2_DATASETS:
        available = ", ".join(sorted(GEOBENCH_V2_DATASETS.keys()))
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {available}"
        )

    local_directory = Path(local_directory)
    repo_id = GEOBENCH_V2_DATASETS[dataset_name]
    files_to_download = GEOBENCH_V2_FILES[dataset_name]
    target_dir = local_directory / dataset_name

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== Downloading {dataset_name} ===")

    for file in files_to_download:
        local_file_path = target_dir / file

        # Skip if file already exists (unless force is enabled)
        if local_file_path.exists() and not force:
            logger.info(f"Skipping {file} (already exists)")
            continue

        logger.info(f"Downloading {file}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=target_dir,
            repo_type="dataset",
        )

    logger.info(f"=== Completed {dataset_name} ===")


def download_geobench_v2(
    local_directory: Path | str,
    datasets: list[str] | None = None,
    force: bool = False,
) -> None:
    """Download GeoBench v2 datasets from Hugging Face.

    Args:
        local_directory: Root directory to download datasets to.
        datasets: List of dataset names to download. If None or empty, downloads all.
        force: Force re-download of files even if they already exist.
    """
    local_directory = Path(local_directory)
    local_directory.mkdir(parents=True, exist_ok=True)

    # If no datasets specified, download all
    if not datasets:
        datasets = list(GEOBENCH_V2_DATASETS.keys())
        logger.info(f"Downloading all GeoBench v2 datasets to: {local_directory}")
    else:
        logger.info(
            f"Downloading {len(datasets)} GeoBench v2 dataset(s) to: {local_directory}"
        )

    for dataset_name in datasets:
        download_geobench_v2_dataset(dataset_name, local_directory, force)

    logger.info("GeoBench v2 downloads completed!")
    logger.info(f"Files downloaded to: {local_directory}")
