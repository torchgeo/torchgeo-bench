#!/usr/bin/env python
"""Analyze class values in GeoBench V2 datasets.

This script inspects all datasets to find:
- Unique class values in labels/masks
- Min/max values
- Potential ignore indices (e.g., 255)
- Number of classes

Useful for configuring ignore_index in segmentation tasks.
"""

import argparse
import logging
import sys
from collections import Counter
from pathlib import Path

import geobench_v2.datasets as gb_v2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset registry with their class names and task types
GEOBENCH_V2_DATASETS = {
    "benv2": {"class": "GeoBenchBENV2", "task": "classification"},
    "biomassters": {"class": "GeoBenchBioMassters", "task": "segmentation"},
    "burn_scars": {"class": "GeoBenchBurnScars", "task": "segmentation"},
    "caffe": {"class": "GeoBenchCaFFe", "task": "segmentation"},
    "cloudsen12": {"class": "GeoBenchCloudSen12", "task": "segmentation"},
    "dynamic_earthnet": {"class": "GeoBenchDynamicEarthNet", "task": "segmentation"},
    "flair2": {"class": "GeoBenchFLAIR2", "task": "segmentation"},
    "forestnet": {"class": "GeoBenchForestnet", "task": "classification"},
    "fotw": {"class": "GeoBenchFieldsOfTheWorld", "task": "segmentation"},
    "kuro_siwo": {"class": "GeoBenchKuroSiwo", "task": "segmentation"},
    "pastis": {"class": "GeoBenchPASTIS", "task": "segmentation"},
    "so2sat": {"class": "GeoBenchSo2Sat", "task": "classification"},
    "spacenet2": {"class": "GeoBenchSpaceNet2", "task": "segmentation"},
    "spacenet7": {"class": "GeoBenchSpaceNet7", "task": "segmentation"},
    "treesatai": {"class": "GeoBenchTreeSatAI", "task": "classification"},
}

SPLITS = ["train", "val", "test"]


def get_dataset_class(dataset_name: str) -> type | None:
    """Get the dataset class from geobench_v2."""
    class_name = GEOBENCH_V2_DATASETS[dataset_name]["class"]
    return getattr(gb_v2, class_name, None)


def analyze_dataset(
    dataset_name: str,
    root: Path,
    max_samples: int | None = None,
    num_workers: int = 0,
    verbose: bool = False,
) -> dict:
    """Analyze class values in a dataset.

    Args:
        dataset_name: Name of the dataset
        root: Root directory containing datasets
        max_samples: Maximum samples to check (None = all)
        num_workers: DataLoader workers
        verbose: Show progress

    Returns:
        Dictionary with analysis results
    """
    if dataset_name not in GEOBENCH_V2_DATASETS:
        return {"status": "error", "message": f"Unknown dataset: {dataset_name}"}

    dataset_dir = root / dataset_name
    if not dataset_dir.exists():
        return {"status": "error", "message": f"Dataset not found: {dataset_dir}"}

    dataset_cls = get_dataset_class(dataset_name)
    if dataset_cls is None:
        return {"status": "error", "message": "Dataset class not found"}

    task_type = GEOBENCH_V2_DATASETS[dataset_name]["task"]

    results = {
        "dataset": dataset_name,
        "task_type": task_type,
        "status": "ok",
        "all_unique_values": set(),
        "value_counts": Counter(),
        "min_value": None,
        "max_value": None,
        "samples_checked": 0,
        "suggested_num_classes": None,
        "suggested_ignore_index": None,
        "splits": {},
    }

    for split in SPLITS:
        split_result = {
            "unique_values": set(),
            "min_value": None,
            "max_value": None,
            "samples_checked": 0,
        }

        try:
            dataset = dataset_cls(root=str(dataset_dir), split=split)
            n_samples = len(dataset)
            if max_samples is not None:
                n_samples = min(n_samples, max_samples)
                dataset = torch.utils.data.Subset(dataset, range(n_samples))

            dataloader = DataLoader(
                dataset,
                batch_size=16,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
            )

            desc = f"{dataset_name}/{split}"
            for batch in tqdm(dataloader, desc=desc, disable=not verbose):
                # Get labels or masks
                if task_type == "classification":
                    if "label" in batch:
                        values = batch["label"]
                    else:
                        continue
                elif task_type == "segmentation":
                    if "mask" in batch:
                        values = batch["mask"]
                    else:
                        continue
                else:
                    continue

                if values is None:
                    continue

                # Convert to tensor if needed
                if isinstance(values, list):
                    values = (
                        torch.stack(values)
                        if all(isinstance(v, torch.Tensor) for v in values)
                        else torch.tensor(values)
                    )

                # Get unique values
                unique = torch.unique(values).tolist()
                split_result["unique_values"].update(unique)
                results["all_unique_values"].update(unique)

                # Update counts
                for val in unique:
                    count = (values == val).sum().item()
                    results["value_counts"][val] += count

                # Track min/max
                val_min = values.min().item()
                val_max = values.max().item()
                if split_result["min_value"] is None or val_min < split_result["min_value"]:
                    split_result["min_value"] = val_min
                if split_result["max_value"] is None or val_max > split_result["max_value"]:
                    split_result["max_value"] = val_max

                split_result["samples_checked"] += values.shape[0]
                results["samples_checked"] += values.shape[0]

        except Exception as e:
            split_result["error"] = str(e)
            logger.error(f"[{dataset_name}/{split}] Error: {e}")

        # Convert set to sorted list for JSON serialization
        split_result["unique_values"] = sorted(split_result["unique_values"])
        results["splits"][split] = split_result

    # Compute overall stats
    all_values = sorted(results["all_unique_values"])
    results["all_unique_values"] = all_values

    if all_values:
        results["min_value"] = min(all_values)
        results["max_value"] = max(all_values)

        # Suggest ignore index and num_classes
        if task_type == "segmentation":
            # Common ignore values: 255, -1, or values far from the main range
            if 255 in all_values:
                results["suggested_ignore_index"] = 255
                valid_classes = [v for v in all_values if v != 255 and v >= 0]
            elif -1 in all_values:
                results["suggested_ignore_index"] = -1
                valid_classes = [v for v in all_values if v != -1]
            else:
                # Check if max value is much larger than others (potential ignore)
                if len(all_values) > 1:
                    sorted_vals = sorted(all_values)
                    if sorted_vals[-1] > sorted_vals[-2] + 100:
                        results["suggested_ignore_index"] = sorted_vals[-1]
                        valid_classes = sorted_vals[:-1]
                    else:
                        valid_classes = all_values
                else:
                    valid_classes = all_values

            if valid_classes:
                results["suggested_num_classes"] = max(valid_classes) + 1
        else:
            # Classification
            results["suggested_num_classes"] = len(all_values)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze class values in GeoBench V2 datasets",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/geobenchv2"),
        help="Root directory (default: data/geobenchv2)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(GEOBENCH_V2_DATASETS.keys()),
        help="Specific datasets to analyze (default: all available)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max samples per split (default: 100, use -1 for all)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        logger.error(f"Root not found: {root}")
        return 1

    # Find available datasets
    available = [d for d in GEOBENCH_V2_DATASETS if (root / d).exists()]
    if not available:
        logger.error(f"No datasets in {root}")
        return 1

    datasets_to_check = args.datasets or available
    datasets_to_check = [d for d in datasets_to_check if d in available]

    if not datasets_to_check:
        logger.error("No matching datasets found")
        return 1

    max_samples = args.max_samples if args.max_samples > 0 else None

    logger.info(f"Analyzing {len(datasets_to_check)} datasets: {datasets_to_check}")

    all_results = {}
    for dataset_name in datasets_to_check:
        logger.info(f"Analyzing {dataset_name}...")
        result = analyze_dataset(
            dataset_name=dataset_name,
            root=root,
            max_samples=max_samples,
            num_workers=args.num_workers,
            verbose=args.verbose,
        )
        all_results[dataset_name] = result

    # Print summary
    print("\n" + "=" * 80)
    print("CLASS VALUE ANALYSIS SUMMARY")
    print("=" * 80)

    for dataset_name, result in all_results.items():
        if result["status"] != "ok":
            print(f"\n✗ {dataset_name}: {result.get('message', 'Error')}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name} ({result['task_type']})")
        print(f"{'=' * 60}")
        print(f"  Samples checked: {result['samples_checked']}")
        print(f"  Unique values: {result['all_unique_values']}")
        print(f"  Min value: {result['min_value']}")
        print(f"  Max value: {result['max_value']}")
        print(f"  Suggested num_classes: {result['suggested_num_classes']}")
        print(f"  Suggested ignore_index: {result['suggested_ignore_index']}")

        # Show per-split details
        for split, split_data in result["splits"].items():
            if "error" in split_data:
                print(f"  {split}: ERROR - {split_data['error']}")
            else:
                print(
                    f"  {split}: values={split_data['unique_values'][:10]}{'...' if len(split_data['unique_values']) > 10 else ''}"
                )

    # Print config suggestions
    print("\n" + "=" * 80)
    print("SUGGESTED CONFIGURATION FOR datasets.py")
    print("=" * 80)
    print("\nNUM_CLASSES_PER_DATASET = {")
    for dataset_name, result in all_results.items():
        if result["status"] == "ok" and result["suggested_num_classes"] is not None:
            print(f'    "{dataset_name}": {result["suggested_num_classes"]},')
    print("}")

    print("\nIGNORE_INDEX_PER_DATASET = {")
    for dataset_name, result in all_results.items():
        if result["status"] == "ok" and result["suggested_ignore_index"] is not None:
            print(f'    "{dataset_name}": {result["suggested_ignore_index"]},')
    print("}")

    # Save to JSON if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=list)
        logger.info(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
