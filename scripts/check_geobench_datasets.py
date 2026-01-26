#!/usr/bin/env python
"""Check that all GeoBench V2 datasets are downloaded correctly.

This script iterates over all samples in each dataset and split to verify
data integrity. It reports any errors encountered during loading.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

import geobench_v2.datasets as gb_v2

# Dataset registry with their class names and task types
GEOBENCH_V2_DATASETS = {
    "benv2": {"class": "GeoBenchBENV2", "task": "classification"},
    "biomassters": {"class": "GeoBenchBioMassters", "task": "segmentation"},
    "burn_scars": {"class": "GeoBenchBurnScars", "task": "segmentation"},
    "caffe": {"class": "GeoBenchCaFFe", "task": "segmentation"},
    "cloudsen12": {"class": "GeoBenchCloudSen12", "task": "segmentation"},
    "dynamic_earthnet": {"class": "GeoBenchDynamicEarthNet", "task": "segmentation"},
    "everwatch": {"class": "GeoBenchEverWatch", "task": "detection"},  # object detection
    "flair2": {"class": "GeoBenchFLAIR2", "task": "segmentation"},
    "forestnet": {"class": "GeoBenchForestnet", "task": "classification"},
    "fotw": {"class": "GeoBenchFieldsOfTheWorld", "task": "segmentation"},
    "kuro_siwo": {"class": "GeoBenchKuroSiwo", "task": "segmentation"},
    "pastis": {"class": "GeoBenchPASTIS", "task": "segmentation"},
    "so2sat": {"class": "GeoBenchSo2Sat", "task": "classification"},
    "spacenet2": {"class": "GeoBenchSpaceNet2", "task": "segmentation"},
    "spacenet7": {"class": "GeoBenchSpaceNet7", "task": "segmentation"},
    "substation": {
        "class": "GeoBenchSubstation",
        "task": "detection",
    },  # instance segmentation with variable boxes
    "treesatai": {"class": "GeoBenchTreeSatAI", "task": "classification"},
}

SPLITS = ["train", "val", "test"]


def detection_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for detection datasets with variable-size boxes.

    Images are stacked, but boxes/labels are kept as lists.
    """
    result: dict[str, Any] = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]
        # Stack tensors that have consistent shapes (like images)
        if isinstance(values[0], torch.Tensor):
            # Check if all have same shape
            shapes = [v.shape for v in values]
            if all(s == shapes[0] for s in shapes):
                result[key] = torch.stack(values)
            else:
                # Keep as list for variable-size tensors (boxes, labels)
                result[key] = values
        else:
            result[key] = values

    return result


def get_dataset_class(dataset_name: str) -> type | None:
    """Get the dataset class from geobench_v2."""
    class_name = GEOBENCH_V2_DATASETS[dataset_name]["class"]
    return getattr(gb_v2, class_name, None)


def check_batch(
    batch: dict[str, Any], batch_idx: int, batch_size: int, task_type: str
) -> list[str]:
    """Check a batch of samples for issues.

    Args:
        batch: The batch dictionary from the DataLoader
        batch_idx: Batch index
        batch_size: Size of the batch
        task_type: 'classification', 'segmentation', or 'detection'

    Returns:
        List of error messages (empty if no errors)
    """
    errors = []
    start_idx = batch_idx * batch_size

    # Find image key (could be 'image', 'image_s2', etc.)
    image_key = None
    for key in batch.keys():
        if key == "image" or key.startswith("image"):
            image_key = key
            break

    if image_key is None:
        errors.append(f"Batch {batch_idx}: Missing image key (no key starting with 'image')")
    else:
        image = batch[image_key]
        if image is None:
            errors.append(f"Batch {batch_idx}: {image_key} is None")
        elif hasattr(image, "shape"):
            # Check for valid dimensions (batch, channels, height, width)
            if len(image.shape) < 3:
                errors.append(f"Batch {batch_idx}: {image_key} has invalid shape {image.shape}")
            # Check for NaN/Inf values
            if torch.isnan(image).any():
                # Find which samples have NaN
                for i in range(image.shape[0]):
                    if torch.isnan(image[i]).any():
                        errors.append(f"Sample {start_idx + i}: {image_key} contains NaN values")
            if torch.isinf(image).any():
                for i in range(image.shape[0]):
                    if torch.isinf(image[i]).any():
                        errors.append(f"Sample {start_idx + i}: {image_key} contains Inf values")

    # Check for label/mask based on task type
    if task_type == "classification":
        if "label" not in batch:
            errors.append(f"Batch {batch_idx}: Missing 'label' key")
    elif task_type == "segmentation":
        if "mask" not in batch:
            errors.append(f"Batch {batch_idx}: Missing 'mask' key")
        else:
            mask = batch["mask"]
            if mask is None:
                errors.append(f"Batch {batch_idx}: Mask is None")
    elif task_type == "detection":
        # Detection tasks have bbox_xyxy and label
        if "bbox_xyxy" not in batch and "boxes" not in batch:
            errors.append(f"Batch {batch_idx}: Missing 'bbox_xyxy' or 'boxes' key")
        if "label" not in batch:
            errors.append(f"Batch {batch_idx}: Missing 'label' key")

    return errors


def check_dataset(
    dataset_name: str,
    root: Path,
    splits: list[str] | None = None,
    max_samples: int | None = None,
    num_workers: int = 0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Check a single dataset for integrity.

    Args:
        dataset_name: Name of the dataset to check
        root: Root directory containing geobench v2 datasets
        splits: List of splits to check (default: all)
        max_samples: Maximum samples to check per split (None = all)
        num_workers: Number of DataLoader workers (default: 0)
        verbose: Print verbose output

    Returns:
        Dictionary with check results
    """
    if dataset_name not in GEOBENCH_V2_DATASETS:
        return {"status": "error", "message": f"Unknown dataset: {dataset_name}"}

    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        return {"status": "error", "message": f"Dataset directory not found: {dataset_dir}"}

    dataset_cls = get_dataset_class(dataset_name)
    if dataset_cls is None:
        return {
            "status": "error",
            "message": f"Dataset class not found: {GEOBENCH_V2_DATASETS[dataset_name]['class']}",
        }

    task_type = GEOBENCH_V2_DATASETS[dataset_name]["task"]
    splits_to_check = splits or SPLITS

    results = {
        "status": "ok",
        "dataset": dataset_name,
        "task_type": task_type,
        "splits": {},
        "total_samples": 0,
        "total_errors": 0,
    }

    for split in splits_to_check:
        split_result = {
            "status": "ok",
            "samples_checked": 0,
            "samples_total": 0,
            "errors": [],
            "load_time": 0.0,
            "check_time": 0.0,
        }

        try:
            # Load the dataset
            start_time = time.time()
            dataset = dataset_cls(
                root=str(dataset_dir),
                split=split,
            )
            split_result["load_time"] = time.time() - start_time
            split_result["samples_total"] = len(dataset)
            results["total_samples"] += len(dataset)

            # Determine how many samples to check
            n_samples = len(dataset)
            if max_samples is not None:
                n_samples = min(n_samples, max_samples)

            # Create DataLoader
            batch_size = 32

            # Use subset if max_samples is set
            if max_samples is not None and max_samples < len(dataset):
                dataset = torch.utils.data.Subset(dataset, range(max_samples))

            # Use custom collate for detection to handle variable-size boxes
            collate_fn = detection_collate_fn if task_type == "detection" else None

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                collate_fn=collate_fn,
            )

            # Iterate through batches
            start_time = time.time()
            desc = f"{dataset_name}/{split}"

            for batch_idx, batch in enumerate(tqdm(dataloader, desc=desc, disable=not verbose)):
                try:
                    errors = check_batch(batch, batch_idx, batch_size, task_type)
                    if errors:
                        split_result["errors"].extend(errors)
                        results["total_errors"] += len(errors)
                except Exception as e:
                    error_msg = f"Batch {batch_idx}: Exception - {type(e).__name__}: {e}"
                    split_result["errors"].append(error_msg)
                    results["total_errors"] += 1
                    if verbose:
                        logger.error(f"[{dataset_name}/{split}] {error_msg}")

                # Count samples in this batch - find first tensor to get batch size
                batch_counted = False
                for key, val in batch.items():
                    if isinstance(val, torch.Tensor) and len(val.shape) >= 1:
                        split_result["samples_checked"] += val.shape[0]
                        batch_counted = True
                        break
                    elif isinstance(val, list):
                        # For detection, values might be lists of tensors
                        split_result["samples_checked"] += len(val)
                        batch_counted = True
                        break
                if not batch_counted:
                    split_result["samples_checked"] += batch_size

            split_result["check_time"] = time.time() - start_time

            if split_result["errors"]:
                split_result["status"] = "errors"
                results["status"] = "errors"

        except Exception as e:
            split_result["status"] = "failed"
            split_result["errors"].append(f"Failed to load dataset: {type(e).__name__}: {e}")
            results["status"] = "errors"
            results["total_errors"] += 1
            logger.error(f"[{dataset_name}/{split}] Failed to load: {e}")

        results["splits"][split] = split_result

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Check GeoBench V2 datasets for integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/geobenchv2"),
        help="Root directory containing geobench v2 datasets (default: data/geobenchv2)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(GEOBENCH_V2_DATASETS.keys()),
        help="Specific datasets to check (default: all available)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        help="Specific splits to check (default: train, val, test)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to check per split (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check: only check first 10 samples per split",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (default: 0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output with progress bars",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        print("Available GeoBench V2 datasets:")
        for name, info in GEOBENCH_V2_DATASETS.items():
            print(f"  {name}: {info['task']}")
        return 0

    # Determine which datasets to check
    root = args.root.resolve()
    if not root.exists():
        logger.error(f"Root directory does not exist: {root}")
        return 1

    # Find available datasets
    available_datasets = []
    for dataset_name in GEOBENCH_V2_DATASETS:
        dataset_dir = root / dataset_name
        if dataset_dir.exists():
            available_datasets.append(dataset_name)

    if not available_datasets:
        logger.error(f"No datasets found in {root}")
        return 1

    # Filter to requested datasets
    datasets_to_check = args.datasets or available_datasets
    datasets_to_check = [d for d in datasets_to_check if d in available_datasets]

    if not datasets_to_check:
        logger.error("No matching datasets found")
        return 1

    # Set max samples for quick mode
    max_samples = args.max_samples
    if args.quick:
        max_samples = 10

    logger.info(f"Checking {len(datasets_to_check)} datasets in {root}")
    if max_samples:
        logger.info(f"Checking up to {max_samples} samples per split")

    # Check each dataset
    all_results = {}
    total_errors = 0
    total_samples = 0

    for dataset_name in datasets_to_check:
        logger.info(f"Checking {dataset_name}...")
        result = check_dataset(
            dataset_name=dataset_name,
            root=root,
            splits=args.splits,
            max_samples=max_samples,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
        all_results[dataset_name] = result
        total_errors += result.get("total_errors", 0)
        total_samples += result.get("total_samples", 0)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for dataset_name, result in all_results.items():
        status_emoji = "✓" if result["status"] == "ok" else "✗"
        print(f"\n{status_emoji} {dataset_name} ({result.get('task_type', 'unknown')})")

        for split, split_result in result.get("splits", {}).items():
            checked = split_result.get("samples_checked", 0)
            total = split_result.get("samples_total", 0)
            n_errors = len(split_result.get("errors", []))
            load_time = split_result.get("load_time", 0)
            check_time = split_result.get("check_time", 0)

            status_icon = "✓" if split_result["status"] == "ok" else "✗"
            print(
                f"  {status_icon} {split}: {checked}/{total} samples checked, "
                f"{n_errors} errors (load: {load_time:.1f}s, check: {check_time:.1f}s)"
            )

            # Print first few errors
            errors = split_result.get("errors", [])
            if errors and args.verbose:
                for error in errors[:5]:
                    print(f"      - {error}")
                if len(errors) > 5:
                    print(f"      ... and {len(errors) - 5} more errors")

    print("\n" + "-" * 60)
    print(f"Total: {total_samples} samples, {total_errors} errors")

    if total_errors > 0:
        logger.warning(f"Found {total_errors} errors across all datasets")
        return 1

    logger.info("All datasets passed integrity check!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
