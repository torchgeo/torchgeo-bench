"""Benchmark ImageStats and three cumulative handcrafted feature levels.

Run from the repository root:
    .venv/bin/python -m experiments.run_handcrafted
    .venv/bin/python -m experiments.run_handcrafted --levels 1 --datasets eurosat

Level zero is the existing ImageStats control. The other levels use the new
handcrafted model. All runs use raw, all-band inputs and the normal benchmark
resize, splits, KNN, linear probe, and bootstrap settings.
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class, list_datasets
from torchgeo_bench.main import resolve_model_config
from torchgeo_bench.resume import _resume_config_hash

from ._runner import Job, add_devices_argument, run_jobs

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]


def classification_datasets() -> list[str]:
    """Return registered image-classification datasets, including multilabel tasks."""
    return [
        name for name in list_datasets() if get_bench_dataset_class(name).task == "classification"
    ]


def model_name(level: int) -> str:
    """Return a distinct result name for each complexity level and the control."""
    return "imagestats_handcrafted_control" if level == 0 else f"handcrafted_level{level}"


def overrides(level: int, dataset: str) -> list[str]:
    """Build benchmark overrides without replacing its evaluation methods."""
    values = [
        f"model={'imagestats' if level == 0 else 'handcrafted'}",
        f"model.name={model_name(level)}",
        f"dataset.names=[{dataset}]",
        "dataset.bands=all",
        "dataset.normalization=identity",
    ]
    if level:
        values.append(f"model.level={level}")
    return values


def build_jobs(level: int, datasets: list[str]) -> list[Job]:
    """Create independent, resumable jobs for one model's classification sweep."""
    return [
        Job(label=f"{model_name(level)}/{name}", overrides=overrides(level, name))
        for name in datasets
    ]


def feature_manifest(level: int, datasets: list[str], devices: list[int]) -> dict:
    """Record each dataset's actual feature columns and sensor inputs."""
    result = {}
    for name in datasets:
        bench = get_bench_dataset_class(name)()
        bands = bench.select_band_specs(None)
        cfg = compose_config(overrides(level, name))
        model_cfg = resolve_model_config(cfg.model, name)
        model = instantiate(model_cfg, bands=bands, normalization="identity")
        if level:
            names = list(model.feature_names)
            metadata = model.feature_metadata
        else:
            names = [
                f"{band.sensor}:{band.name}:{stat}"
                for stat in ("mean", "std", "max", "min")
                for band in bands
            ]
            metadata = {}
        result[name] = {
            "bands": [asdict(band) for band in bands],
            "feature_dim": len(names),
            "feature_names": names,
            "feature_metadata": metadata,
            "num_classes": bench.num_classes,
            "multilabel": bench.multilabel,
            "split_sizes": bench.split_sizes,
            "config_hashes": {
                _resume_config_hash(
                    compose_config([*overrides(level, name), f"device=cuda:{device}"])
                ): f"cuda:{device}"
                for device in devices
            },
        }
    return result


def completed_rows(path: Path) -> list[dict]:
    """Read result rows if the model has produced an output file."""
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def validate_case(rows: list[dict], name: str, spec: dict) -> list[str]:
    """Check both expected methods, dimensions, labels, and split counts."""
    failures = []
    metric = "micro_mAP" if spec["multilabel"] else "accuracy"
    for method in ("knn5", "linear"):
        selected = [
            row
            for row in rows
            if row["dataset"] == name
            and row["method"] == method
            and row["config_hash"] in spec["config_hashes"]
        ]
        if len(selected) != 1:
            failures.append(f"{name}/{method}: expected one row, found {len(selected)}")
            continue
        row = selected[0]
        valid = (
            row["metric_name"] == metric
            and row["bands"] == "all"
            and row["normalization"] == "identity"
            and row["merge_val"].lower() == "true"
            and int(float(row["feature_dim"])) == spec["feature_dim"]
            and int(float(row["num_classes"])) == spec["num_classes"]
            and float(row["image_size"]) == 224
            and int(float(row["seed"])) == 0
            and row["interpolation"] == "bilinear"
            and row["partition"] == "default"
            and int(float(row["bootstrap"])) == 200
            and float(row["c_range_start"]) == -6
            and float(row["c_range_stop"]) == 4
            and int(float(row["c_range_num"])) == 40
        )
        for split, count in spec["split_sizes"].items():
            valid = valid and int(float(row[f"n_{split}"])) == count
        score = float(row["metric_value"])
        low, high = float(row["ci_lower"]), float(row["ci_upper"])
        valid = valid and all(math.isfinite(value) for value in (score, low, high))
        valid = valid and 0 <= score <= 1 and 0 <= low <= high <= 1
        if not valid:
            failures.append(f"{name}/{method}: result settings or score do not match the sweep")
    return failures


def save_json(path: Path, value: dict) -> None:
    """Write a complete JSON artifact with atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_level(args: argparse.Namespace, level: int) -> bool:
    """Run one complexity level and verify that both methods actually completed."""
    name = model_name(level)
    manifest = feature_manifest(level, args.datasets, args.devices)
    save_json(args.run_dir / f"{name}.features.json", manifest)
    output = args.output_dir / f"{name}.csv"
    code = run_jobs(
        build_jobs(level, args.datasets), args.devices, output=str(output), dry_run=args.dry_run
    )
    if args.dry_run:
        return code == 0
    rows = completed_rows(output)
    failures = [
        message
        for dataset in args.datasets
        for message in validate_case(rows, dataset, manifest[dataset])
    ]
    status = {
        "level": level,
        "output": str(output),
        "devices": args.devices,
        "exit_code": code,
        "failures": failures,
    }
    save_json(args.run_dir / f"{name}.status.json", status)
    for message in failures:
        logger.error("%s", message)
    return code == 0 and not failures


def summarize(args: argparse.Namespace) -> None:
    """Write scores by dataset and level without averaging different task metrics."""
    baseline = {
        (row["dataset"], row["method"]): float(row["metric_value"])
        for row in completed_rows(args.output_dir / f"{model_name(0)}.csv")
    }
    summary = []
    for level in args.levels:
        for row in completed_rows(args.output_dir / f"{model_name(level)}.csv"):
            if row["dataset"] not in args.datasets:
                continue
            control = baseline.get((row["dataset"], row["method"]))
            score = float(row["metric_value"])
            summary.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "metric": row["metric_name"],
                    "level": level,
                    "feature_dim": row["feature_dim"],
                    "config_hash": row["config_hash"],
                    "score": score,
                    "ci_lower": row["ci_lower"],
                    "ci_upper": row["ci_upper"],
                    "change_from_imagestats": score - control if control is not None else "",
                    "best_c": row.get("best_c", ""),
                }
            )
    if not summary:
        raise ValueError("No result rows are available for the requested sweep.")
    name = f"summary_level{args.levels[0]}.csv" if len(args.levels) == 1 else "summary.csv"
    with (args.run_dir / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


def main() -> int:
    """Run the requested classification sweep, retaining explicit failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", nargs="+", type=int, choices=(0, 1, 2, 3), default=[0, 1, 2, 3])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/models")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs/handcrafted")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    add_devices_argument(parser)
    args = parser.parse_args()
    available = classification_datasets()
    args.datasets = available if args.datasets is None else args.datasets
    if not args.datasets or not set(args.datasets).issubset(available):
        parser.error("--datasets must contain registered image-classification datasets")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.levels)) != len(args.levels):
        parser.error("datasets and levels must be unique")
    if not args.devices or len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("--devices must contain distinct nonnegative GPU indices")
    if (
        not args.dry_run
        and not args.report_only
        and (not torch.cuda.is_available() or max(args.devices) >= torch.cuda.device_count())
    ):
        parser.error("a requested CUDA device is unavailable")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.report_only:
        summarize(args)
        return 0
    passed = [run_level(args, level) for level in args.levels]
    if not args.dry_run:
        summarize(args)
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
