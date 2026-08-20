"""Run label-informed spatial priors on the CoordBench classification tasks."""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from rich.progress import track

from torchgeo_bench.coordbench.baselines import (
    ClassFrequencyPrior,
    GridPrior,
    KDEPrior,
    NearestNeighborPrior,
    SpatialPrior,
    UniformPrior,
)
from torchgeo_bench.coordbench.datasets import CoordBenchmark, load_benchmarks
from torchgeo_bench.coordbench.probe import _valid_mask, spatial_fold_ids

logger = logging.getLogger(__name__)

RESUME_KEY_COLS = ("dataset", "task", "method", "model_name", "split")
PRIOR_NAMES = ("uniform", "frequency", "grid", "nearest", "kde")


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

    def to_row(self) -> dict:
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


def _make_prior(name: str, cfg: DictConfig) -> SpatialPrior:
    """Construct a configured spatial prior by its public short name."""
    if name == "uniform":
        return UniformPrior()
    if name == "frequency":
        return ClassFrequencyPrior()
    if name == "grid":
        return GridPrior(
            cell_size=float(cfg.grid_cell_size),
            smoothing=float(cfg.smoothing),
        )
    if name == "nearest":
        return NearestNeighborPrior(
            n_neighbors=int(cfg.nearest_k),
            weights=str(cfg.nearest_weights),
        )
    if name == "kde":
        return KDEPrior(
            bandwidth=float(cfg.kde_bandwidth),
            smoothing=float(cfg.smoothing),
        )
    raise ValueError(f"Unknown coord_prior method {name!r}; choose from {PRIOR_NAMES}")


def _resolve_splits(split: str) -> list[str]:
    """Expand the configured split selector."""
    if split == "both":
        return ["random", "spatial"]
    if split not in ("random", "spatial"):
        raise ValueError("coord_prior.split must be one of random|spatial|both")
    return [split]


def _fold_pairs(
    n_samples: int,
    folds: int,
    seed: int,
    fold_assign: np.ndarray | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return train/test index pairs for random or spatial CV."""
    if fold_assign is None:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(n_samples, generator=generator).numpy()
        test_folds = [order[i::folds] for i in range(folds)]
    else:
        test_folds = [np.flatnonzero(fold_assign == f) for f in np.unique(fold_assign)]
    pairs = []
    for i, test in enumerate(test_folds):
        train_parts = test_folds[:i] + test_folds[i + 1 :]
        if len(test) == 0 or not train_parts:
            continue
        pairs.append((np.concatenate(train_parts), test))
    return pairs


def _score_prior(
    name: str,
    cfg: DictConfig,
    lon: np.ndarray,
    lat: np.ndarray,
    labels: np.ndarray,
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """Fit and score a fresh prior independently on each holdout split."""
    scores = []
    for train_idx, test_idx in pairs:
        fitted = _make_prior(name, cfg)
        fitted.fit(lon[train_idx], lat[train_idx], labels[train_idx])
        probabilities = fitted.predict_proba(lon[test_idx], lat[test_idx])
        predictions = fitted.classes_[probabilities.argmax(axis=1)]
        scores.append(float(np.mean(predictions == labels[test_idx])))
    return scores


def _evaluate_benchmark(
    bench: CoordBenchmark,
    *,
    methods: Sequence[str],
    splits: Sequence[str],
    cfg: DictConfig,
    seed: int,
    model_target: dict[str, str],
    completed: set[tuple[str, ...]] | None,
) -> list[dict]:
    """Evaluate all selected priors on one classification benchmark."""
    if bench.task_type != "classification":
        return []

    rows: list[dict] = []
    for task, raw_labels in bench.tasks.items():
        labels = np.asarray(raw_labels)
        valid = _valid_mask(np.ones((len(labels), 1), dtype=np.float32), labels, "classification")
        lon, lat, labels = bench.lon[valid], bench.lat[valid], labels[valid]
        if len(labels) < 2:
            continue

        for split in splits:
            if bench.test_mask is not None:
                test_mask = np.asarray(bench.test_mask, dtype=bool)[valid]
                pairs = [(np.flatnonzero(~test_mask), np.flatnonzero(test_mask))]
                split_label = "official"
            elif split == "spatial":
                fold_assign = spatial_fold_ids(lat, lon, int(cfg.folds), float(cfg.cell_deg), seed)
                pairs = _fold_pairs(len(labels), int(cfg.folds), seed, fold_assign)
                split_label = "spatial"
            else:
                pairs = _fold_pairs(len(labels), int(cfg.folds), seed, None)
                split_label = "random"

            if not pairs:
                continue
            n_test = len(pairs[0][1]) if split_label == "official" else len(labels)
            for name in methods:
                key = (bench.name, task, "prior", name, split_label)
                if completed is not None and tuple(map(str, key)) in completed:
                    continue
                scores = _score_prior(name, cfg, lon, lat, labels, pairs)
                score = float(np.mean(scores))
                spread = float(np.std(scores)) if len(scores) > 1 else 0.0
                rows.append(
                    CoordPriorResult(
                        dataset=bench.name,
                        task=task,
                        task_type=bench.task_type,
                        method="prior",
                        split=split_label,
                        metric_name="accuracy",
                        metric_value=score,
                        ci_lower=score - spread,
                        ci_upper=score + spread,
                        n_folds=len(scores),
                        cell_deg=float(cfg.cell_deg),
                        feature_dim=0,
                        n_samples=len(raw_labels),
                        n_test=n_test,
                        seed=seed,
                        model_name=name,
                        model_target=model_target[name],
                    ).to_row()
                )
            if bench.test_mask is not None:
                break
    return rows


def run_coordbench_priors(cfg: DictConfig) -> None:
    """Run label-informed spatial priors on CoordBench classification tasks."""
    from torchgeo_bench.main import append_rows_atomic

    prior_cfg = cfg.coord_prior
    methods = [str(name) for name in prior_cfg.methods]
    unknown = sorted(set(methods) - set(PRIOR_NAMES))
    if unknown:
        raise ValueError(f"Unknown coord_prior methods: {', '.join(unknown)}")
    splits = _resolve_splits(str(prior_cfg.split))
    output_path = str(prior_cfg.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    model_target = {
        name: f"torchgeo_bench.coordbench.baselines.{type(_make_prior(name, prior_cfg)).__name__}"
        for name in methods
    }
    completed = _completed_keys(output_path) if cfg.resume else None
    benchmarks = load_benchmarks(prior_cfg.names)
    logger.info("CoordBench priors: %d benchmarks, methods=%s", len(benchmarks), methods)

    for bench in track(benchmarks, description="CoordBench priors"):
        rows = _evaluate_benchmark(
            bench,
            methods=methods,
            splits=splits,
            cfg=prior_cfg,
            seed=int(cfg.seed),
            model_target=model_target,
            completed=completed,
        )
        if rows:
            append_rows_atomic(output_path, rows)

    logger.info("CoordBench prior baselines complete. Results appended to %s", output_path)
