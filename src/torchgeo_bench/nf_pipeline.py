"""Stage-1 Normalizing Flow pipeline: Optuna HPO + test evaluation."""

from __future__ import annotations

import logging
import os
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.main import append_rows_atomic
from torchgeo_bench.uq.metrics import brier_score, ece, nll
from torchgeo_bench.utils import extract_features

logger = logging.getLogger(__name__)

_METRICS = frozenset({"accuracy", "nll", "ece", "brier"})


def _is_done(df: pd.DataFrame | None, model: str, name: str, dataset: str,
             partition: str, bands: str, seed: int) -> bool:
    if df is None or df.empty:
        return False
    mask = (
        (df["model"] == model)
        & (df["name"] == name)
        & (df["dataset"] == dataset)
        & (df["partition"] == partition)
        & (df["bands"] == bands)
        & (df["seed"] == seed)
    )
    existing = set(df.loc[mask, "metric_name"].tolist())
    return _METRICS.issubset(existing)


def _load_existing(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def _evaluate(probe: Any, X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
               n_bootstrap: int) -> dict[str, float]:
    probs = probe.predict_proba(X)
    y_pred = probs.argmax(axis=1)
    point_acc = float((y_pred == y).mean())
    point_nll = float(nll(probs, y))
    point_ece = float(ece(probs, y, n_bins=15, binning="equal_width"))
    point_brier = float(brier_score(probs, y))

    if n_bootstrap <= 1:
        return {"accuracy": point_acc, "nll": point_nll, "ece": point_ece, "brier": point_brier}

    accs, nlls, eces, briers = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), size=len(y))
        p_b, y_b = probs[idx], y[idx]
        y_pred_b = p_b.argmax(axis=1)
        accs.append(float((y_pred_b == y_b).mean()))
        nlls.append(float(nll(p_b, y_b)))
        eces.append(float(ece(p_b, y_b, n_bins=15, binning="equal_width")))
        briers.append(float(brier_score(p_b, y_b)))

    return {
        "accuracy": float(np.mean(accs)),
        "nll": float(np.mean(nlls)),
        "ece": float(np.mean(eces)),
        "brier": float(np.mean(briers)),
    }


@hydra.main(config_path="conf", config_name="nf_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the NF stage-1 pipeline.

    Args:
        cfg: Hydra configuration.
    """
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("optuna>=3.0 required; pip install 'torchgeo-bench[uq]'") from exc

    from torchgeo_bench.uq.nf import NormalizingFlowProbe

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    output_path = str(cfg.nf.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    device = torch.device(str(cfg.device))
    model_target = str(cfg.model._target_)
    model_name = str(cfg.model.get("name", model_target.split(".")[-1]))
    dataset_names = list(cfg.dataset.names)
    n_trials = int(cfg.nf.n_trials)
    n_bootstrap = int(cfg.nf.bootstrap)
    epochs = int(cfg.nf.epochs)
    batch_size = int(cfg.nf.batch_size)
    partition = str(cfg.dataset.partition)
    bands = str(cfg.dataset.bands)
    seed = int(cfg.seed)

    existing_df = _load_existing(output_path) if bool(cfg.resume) else None

    for dataset_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(dataset_name)
        except KeyError:
            logger.warning("Skipping unknown dataset %s", dataset_name)
            continue

        if ds_cls.task != "classification" or getattr(ds_cls, "multilabel", False):
            logger.info("Skipping non-classification dataset %s", dataset_name)
            continue

        if bool(cfg.resume) and _is_done(existing_df, model_target, model_name,
                                          dataset_name, partition, bands, seed):
            logger.info("Already done: %s / %s — skipping.", model_name, dataset_name)
            continue

        logger.info("Processing %s / %s", model_name, dataset_name)

        # Feature extraction
        loaded = get_datasets(
            dataset_name=dataset_name,
            partition_name=partition,
            batch_size=int(cfg.dataset.batch_size),
            num_workers=int(cfg.dataset.get("num_workers", 4)),
            return_val=True,
            image_size=getattr(cfg.dataset, "image_size", None),
            interpolation=str(cfg.dataset.get("interpolation", "bilinear")),
            bands=bands,
        )
        train_loader, val_loader, test_loader, _ = loaded
        model = instantiate(cfg.model)

        X_train, y_train = extract_features(model, train_loader, device, verbose=bool(cfg.verbose))
        X_val, y_val = extract_features(model, val_loader, device, verbose=bool(cfg.verbose))
        X_test, y_test = extract_features(model, test_loader, device, verbose=bool(cfg.verbose))

        # Val split from train if no explicit val loader
        if X_val is None or len(X_val) == 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train
            )

        X_tr, y_tr, X_v, y_v = X_train, y_train, X_val, y_val

        def _objective(trial: Any, X_tr: np.ndarray = X_tr, y_tr: np.ndarray = y_tr,
                       X_v: np.ndarray = X_v, y_v: np.ndarray = y_v) -> float:
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            wd = trial.suggest_float("wd", 1e-5, 1e-1, log=True)
            probe = NormalizingFlowProbe(
                prior="empirical", lr=lr, weight_decay=wd,
                epochs=epochs, batch_size=batch_size,
            )
            probe.fit(X_tr, y_tr)
            val_probs = probe.predict_proba(X_v)
            return float(nll(val_probs, y_v))

        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(_objective, n_trials=n_trials)

        best_lr = float(study.best_params["lr"])
        best_wd = float(study.best_params["wd"])
        val_nll = float(study.best_value)

        # Refit on full train with best hyperparams
        best_probe = NormalizingFlowProbe(
            prior="empirical", lr=best_lr, weight_decay=best_wd,
            epochs=epochs, batch_size=batch_size,
        )
        best_probe.fit(X_train, y_train)
        metrics = _evaluate(best_probe, X_test, y_test, rng, n_bootstrap)

        rows = [
            {
                "model": model_target,
                "name": model_name,
                "dataset": dataset_name,
                "partition": partition,
                "bands": bands,
                "seed": seed,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "best_lr": best_lr,
                "best_wd": best_wd,
                "val_nll": val_nll,
                "n_trials": n_trials,
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
            }
            for metric_name, metric_value in metrics.items()
        ]
        append_rows_atomic(output_path, rows)
        logger.info("Wrote %d rows for %s / %s", len(rows), model_name, dataset_name)
