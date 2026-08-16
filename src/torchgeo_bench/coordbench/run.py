"""Runner for the CoordBench location-encoder track.

Driven from ``torchgeo-bench run mode=coord``: instantiate a coordinate encoder
from the Hydra ``model`` config, embed each benchmark's points once, then probe
with KNN and/or a ridge linear head under random and/or spatial-block
cross-validation. One CSV row per (benchmark, task, method, split) is appended
to ``coord.output`` via the shared atomic writer, with resume support.
"""

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from torchgeo_bench.config import instantiate
from torchgeo_bench.coordbench.datasets import CoordBenchmark, load_benchmarks
from torchgeo_bench.coordbench.models import LocationEncoder
from torchgeo_bench.coordbench.probe import (
    knn_probe_score,
    linear_probe_score,
    spatial_fold_ids,
)

logger = logging.getLogger(__name__)

RESUME_KEY_COLS = ("dataset", "task", "method", "model_name", "split")


@dataclass
class CoordResult:
    """A single CoordBench evaluation result row."""

    dataset: str  # benchmark name (e.g. "sustainbench-asset")
    task: str  # label/column name within the benchmark
    task_type: str  # "regression" | "classification"
    method: str  # "linear" | "knn{k}"
    split: str  # "random" | "spatial" | "official"
    metric_name: str  # "r2" | "accuracy"
    metric_value: float  # mean over folds (or the held-out score)
    ci_lower: float  # metric_value - std over folds
    ci_upper: float  # metric_value + std over folds
    n_folds: int
    cell_deg: float
    feature_dim: int
    n_samples: int
    n_test: int
    seed: int
    model_name: str
    model_target: str

    def to_row(self) -> dict:
        """Convert to a flat dict suitable for CSV/DataFrame export."""
        return self.__dict__.copy()


def _instantiate_encoder(model_cfg: DictConfig, device: str) -> tuple[LocationEncoder, str]:
    """Build a :class:`LocationEncoder` from a Hydra model config.

    Returns the encoder and the ``name`` recorded in result rows. ``name`` is a
    display field, not a constructor argument, so it is stripped before
    instantiation.
    """
    cfg = model_cfg.copy()
    OmegaConf.set_struct(cfg, False)
    name = cfg.pop("name", cfg.get("_target_", "model").split(".")[-1])
    encoder = instantiate(cfg, device=device)
    if not isinstance(encoder, LocationEncoder):
        raise TypeError(
            f"mode=coord requires model._target_ to be a LocationEncoder subclass; "
            f"got {type(encoder).__name__}. Pick a coord model, e.g. model=sincos."
        )
    return encoder, str(name)


def _resolve_splits(split: str) -> list[str]:
    """Expand the ``coord.split`` value into concrete CV modes to run."""
    if split == "both":
        return ["random", "spatial"]
    if split not in ("random", "spatial"):
        raise ValueError(f"coord.split must be one of random|spatial|both; got {split!r}")
    return [split]


def _completed_keys(output_path: str) -> set[tuple[str, ...]]:
    """Existing (dataset, task, method, model_name, split) keys for resume."""
    if not os.path.exists(output_path):
        return set()
    df = pd.read_csv(output_path)
    for col in RESUME_KEY_COLS:
        if col not in df.columns:
            return set()
    rows = df[list(RESUME_KEY_COLS)].fillna("").astype(str).to_numpy()
    return {tuple(r) for r in rows}


def _methods_for(task_type: str, requested: Sequence[str], knn_k: int) -> list[tuple[str, str]]:
    """Resolve (method-label, kind) pairs applicable to a task type.

    KNN is classification-only (there is no KNN-regression head here); the ridge
    linear probe handles both regression (R^2) and classification (accuracy).
    """
    methods: list[tuple[str, str]] = []
    if "knn" in requested and task_type == "classification":
        methods.append((f"knn{knn_k}", "knn"))
    if "linear" in requested:
        methods.append(("linear", "linear"))
    return methods


def _score_one(
    kind: str,
    features: np.ndarray,
    labels: np.ndarray,
    task_type: str,
    *,
    folds: int,
    seed: int,
    device: str,
    knn_device: str,
    knn_k: int,
    test_mask: np.ndarray | None,
    fold_assign: np.ndarray | None,
) -> tuple[float, list[float]]:
    if kind == "knn":
        return knn_probe_score(
            features,
            labels,
            folds=folds,
            seed=seed,
            k=knn_k,
            device=knn_device,
            test_mask=test_mask,
            fold_assign=fold_assign,
        )
    return linear_probe_score(
        features,
        labels,
        task_type,
        folds=folds,
        seed=seed,
        device=device,
        test_mask=test_mask,
        fold_assign=fold_assign,
    )


def run_coordbench(cfg: DictConfig) -> None:
    """Run the CoordBench location-encoder benchmark for the configured model."""
    from torchgeo_bench.main import append_rows_atomic  # lazy: avoids import cycle

    coord = cfg.coord
    device = str(cfg.device)
    seed = int(cfg.seed)
    folds = int(coord.folds)
    cell_deg = float(coord.cell_deg)
    knn_k = int(coord.knn_k)
    knn_device = str(coord.get("knn_device") or "cpu")
    methods = list(coord.methods)
    splits = _resolve_splits(str(coord.split))

    output_path = str(coord.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    encoder, model_name = _instantiate_encoder(cfg.model, device)
    model_target = str(cfg.model.get("_target_", type(encoder).__name__))
    logger.info("CoordBench: model=%s device=%s splits=%s", model_name, device, splits)

    completed = _completed_keys(output_path) if cfg.resume else set()
    if completed:
        logger.info("Resume mode: %d existing coord results in %s", len(completed), output_path)

    benchmarks = load_benchmarks(coord.names)
    logger.info("CoordBench: %d benchmarks selected", len(benchmarks))

    for bench in track(benchmarks, description="CoordBench"):
        rows = _evaluate_benchmark(
            bench,
            encoder,
            methods=methods,
            splits=splits,
            folds=folds,
            cell_deg=cell_deg,
            knn_k=knn_k,
            knn_device=knn_device,
            seed=seed,
            device=device,
            model_name=model_name,
            model_target=model_target,
            completed=completed if cfg.resume else None,
        )
        if rows:
            append_rows_atomic(output_path, rows)

    logger.info("CoordBench complete. Results appended to %s", output_path)


def _evaluate_benchmark(
    bench: CoordBenchmark,
    encoder: LocationEncoder,
    *,
    methods: Sequence[str],
    splits: Sequence[str],
    folds: int,
    cell_deg: float,
    knn_k: int,
    knn_device: str,
    seed: int,
    device: str,
    model_name: str,
    model_target: str,
    completed: set[tuple[str, ...]] | None,
) -> list[dict]:
    """Embed one benchmark once and probe every (task, method, split) combination."""
    metric_name = "r2" if bench.task_type == "regression" else "accuracy"
    method_kinds = _methods_for(bench.task_type, methods, knn_k)
    if not method_kinds:
        return []

    features = encoder.encode(bench.lon, bench.lat, bench.year)
    feature_dim = int(features.shape[1])

    rows: list[dict] = []
    for split in splits:
        # Official held-out split wins when present; else the requested CV mode.
        if bench.test_mask is not None:
            test_mask, fold_assign, split_label = bench.test_mask, None, "official"
        elif split == "spatial":
            test_mask, fold_assign, split_label = (
                None,
                spatial_fold_ids(bench.lat, bench.lon, folds, cell_deg, seed),
                "spatial",
            )
        else:
            test_mask, fold_assign, split_label = None, None, "random"

        for task, labels in bench.tasks.items():
            for method_label, kind in method_kinds:
                key = (bench.name, task, method_label, model_name, split_label)
                if completed is not None and tuple(map(str, key)) in completed:
                    continue
                score, fold_scores = _score_one(
                    kind,
                    features,
                    np.asarray(labels),
                    bench.task_type,
                    folds=folds,
                    seed=seed,
                    device=device,
                    knn_device=knn_device,
                    knn_k=knn_k,
                    test_mask=test_mask,
                    fold_assign=fold_assign,
                )
                std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0
                if test_mask is not None:
                    n_test = int(np.asarray(test_mask, dtype=bool).sum())
                elif bench.task_type == "regression":
                    n_test = int(np.isfinite(np.asarray(labels, dtype=np.float64)).sum())
                else:
                    n_test = int(len(labels))
                rows.append(
                    CoordResult(
                        dataset=bench.name,
                        task=task,
                        task_type=bench.task_type,
                        method=method_label,
                        split=split_label,
                        metric_name=metric_name,
                        metric_value=score,
                        ci_lower=score - std,
                        ci_upper=score + std,
                        n_folds=1 if split_label == "official" else folds,
                        cell_deg=cell_deg,
                        feature_dim=feature_dim,
                        n_samples=len(labels),
                        n_test=n_test,
                        seed=seed,
                        model_name=model_name,
                        model_target=model_target,
                    ).to_row()
                )
        # A benchmark with an official split is split-invariant; don't re-run per CV mode.
        if bench.test_mask is not None:
            break
    return rows
