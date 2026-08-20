"""Run label-informed spatial priors on the CoordBench classification tasks."""

import logging
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torchgeo_bench.coordbench.baselines import (
    ClassFrequencyPrior,
    GridPrior,
    KDEPrior,
    NearestNeighborPrior,
    SpatialPrior,
    UniformPrior,
)
from torchgeo_bench.coordbench.datasets import CoordBenchmark, load_benchmarks
from torchgeo_bench.coordbench.prior_config import (
    PRIOR_NAMES,
    CoordPriorConfig,
    CoordPriorEvaluationConfig,
)
from torchgeo_bench.coordbench.probe import _fold_indices, _valid_mask, spatial_fold_ids
from torchgeo_bench.results import append_rows_atomic

logger = logging.getLogger(__name__)

RESUME_KEY_COLS = ("dataset", "task", "method", "model_name", "split")


@dataclass
class CoordPriorResult:
    """One classification-prior result row."""

    dataset: str
    task: str
    task_type: str
    method: str
    split: str
    metric_name: str
    metric_value: float
    ci_lower: float
    ci_upper: float
    n_folds: int
    cell_deg: float
    feature_dim: int
    n_samples: int
    n_test: int
    seed: int
    model_name: str
    model_target: str

    def to_row(self) -> dict[str, Any]:
        """Convert the result to a CSV-compatible mapping."""
        return self.__dict__.copy()


def _completed_keys(output_path: str) -> set[tuple[str, ...]]:
    """Return existing prior result keys for resume mode."""
    if not os.path.exists(output_path):
        return set()
    df = pd.read_csv(output_path)
    if any(col not in df.columns for col in RESUME_KEY_COLS):
        return set()
    rows = df[list(RESUME_KEY_COLS)].fillna("").astype(str).to_numpy()
    return {tuple(row) for row in rows}


def _make_prior(name: str, cfg: CoordPriorEvaluationConfig) -> SpatialPrior:
    """Construct a configured spatial prior by its public short name."""
    if name == "uniform":
        return UniformPrior()
    if name == "frequency":
        return ClassFrequencyPrior()
    if name == "grid":
        return GridPrior(
            cell_size=cfg.grid_cell_size,
            smoothing=cfg.smoothing,
        )
    if name == "nearest":
        return NearestNeighborPrior(
            n_neighbors=cfg.nearest_k,
            weights=cfg.nearest_weights,
        )
    if name == "kde":
        return KDEPrior(
            bandwidth=cfg.kde_bandwidth,
            smoothing=cfg.smoothing,
        )
    raise ValueError(f"Unknown prior method {name!r}; choose from {PRIOR_NAMES}")


def _resolve_splits(split: str) -> list[str]:
    """Expand the configured split selector."""
    if split == "both":
        return ["random", "spatial"]
    if split not in ("random", "spatial"):
        raise ValueError("evaluation.split must be one of random|spatial|both")
    return [split]


def _fold_pairs(
    n_samples: int,
    folds: int,
    seed: int,
    fold_assign: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test index pairs for random or spatial CV."""
    test_folds = _fold_indices(n_samples, folds, seed, fold_assign)
    pairs = []
    for i, test in enumerate(test_folds):
        train_parts = test_folds[:i] + test_folds[i + 1 :]
        if len(test) == 0 or not train_parts:
            continue
        train = np.concatenate(train_parts)
        if len(train):
            pairs.append((train, test))
    return pairs


def _evaluation_pairs(
    bench: CoordBenchmark, valid: np.ndarray, cfg: CoordPriorConfig, split: str
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Select valid official examples or construct the requested CV folds."""
    if bench.test_mask is not None:
        test_mask = np.asarray(bench.test_mask, dtype=bool)[valid]
        train, test = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
        return [(train, test)] if len(train) and len(test) else []
    fold_assign = None
    if split == "spatial":
        fold_assign = spatial_fold_ids(
            bench.lat[valid],
            bench.lon[valid],
            cfg.evaluation.folds,
            cfg.evaluation.cell_deg,
            cfg.runtime.seed,
        )
    return _fold_pairs(int(valid.sum()), cfg.evaluation.folds, cfg.runtime.seed, fold_assign)


def _score_prior(
    name: str,
    cfg: CoordPriorEvaluationConfig,
    coordinates: np.ndarray,
    labels: np.ndarray,
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Fit and score a fresh prior independently on each holdout split."""
    scores = []
    for train_idx, test_idx in pairs:
        fitted = _make_prior(name, cfg)
        fitted.fit(coordinates[train_idx, 0], coordinates[train_idx, 1], labels[train_idx])
        probabilities = fitted.predict_proba(coordinates[test_idx, 0], coordinates[test_idx, 1])
        predictions = fitted.classes_[probabilities.argmax(axis=1)]
        scores.append(float(np.mean(predictions == labels[test_idx])))
    return scores


def _evaluate_benchmark(
    bench: CoordBenchmark,
    cfg: CoordPriorConfig,
    completed: set[tuple[str, ...]],
) -> Iterator[dict[str, Any]]:
    """Evaluate all selected priors on one classification benchmark."""
    if bench.task_type != "classification":
        logger.info("Skipping %s: spatial priors support classification only", bench.name)
        return

    splits = ["official"] if bench.test_mask is not None else _resolve_splits(cfg.evaluation.split)
    coordinates = np.column_stack((bench.lon, bench.lat))
    for task, raw_labels in bench.tasks.items():
        labels = np.asarray(raw_labels)
        valid = (
            _valid_mask(coordinates, labels, "classification")
            if len(labels)
            else np.zeros(0, dtype=bool)
        )
        for split in splits:
            pairs = _evaluation_pairs(bench, valid, cfg, split)
            if not pairs:
                logger.warning(
                    "Skipping %s/%s (%s): no non-empty valid training/test split",
                    bench.name,
                    task,
                    split,
                )
                continue
            for name in cfg.evaluation.methods:
                key = (bench.name, task, "prior", name, split)
                if key in completed:
                    continue
                scores = _score_prior(
                    name, cfg.evaluation, coordinates[valid], labels[valid], pairs
                )
                score = float(np.mean(scores))
                spread = float(np.std(scores)) if len(scores) > 1 else 0.0
                prior_class = type(_make_prior(name, cfg.evaluation)).__name__
                yield CoordPriorResult(
                    dataset=bench.name,
                    task=task,
                    task_type=bench.task_type,
                    method="prior",
                    split=split,
                    metric_name="accuracy",
                    metric_value=score,
                    ci_lower=score - spread,
                    ci_upper=score + spread,
                    n_folds=len(scores),
                    cell_deg=cfg.evaluation.cell_deg,
                    feature_dim=0,
                    n_samples=len(raw_labels),
                    n_test=sum(len(test) for _, test in pairs),
                    seed=cfg.runtime.seed,
                    model_name=name,
                    model_target=f"torchgeo_bench.coordbench.baselines.{prior_class}",
                ).to_row()


def run_coordbench_priors(cfg: CoordPriorConfig) -> None:
    """Run label-informed spatial priors on CoordBench classification tasks."""
    output_path = cfg.output.file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    completed = _completed_keys(output_path) if cfg.output.resume else set()
    names = "all" if cfg.datasets == ["all"] else cfg.datasets
    benchmarks = load_benchmarks(names)
    logger.info(
        "CoordBench priors: %d benchmarks, methods=%s", len(benchmarks), cfg.evaluation.methods
    )

    for bench in tqdm(benchmarks, desc="CoordBench priors"):
        for row in _evaluate_benchmark(bench, cfg, completed):
            append_rows_atomic(output_path, [row])
            completed.add(tuple(str(row[col]) for col in RESUME_KEY_COLS))

    logger.info("CoordBench prior baselines complete. Results appended to %s", output_path)
