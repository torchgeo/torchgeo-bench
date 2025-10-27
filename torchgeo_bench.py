"""Benchmark script for torchgeo-bench."""

import os
import io
import fcntl
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.linear import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from src.datasets import NUM_CLASSES_PER_DATASET, get_datasets
from src.models.interface import BenchModel
from src.utils import extract_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _expand_dataset_list(names):
    if isinstance(names, str):
        if names == "all":
            return list(NUM_CLASSES_PER_DATASET.keys())
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs = np.empty(n_boot, dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        accs[i] = (y_true[idx] == y_pred[idx]).mean()
    acc_mean = float((y_true == y_pred).mean())
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = float(np.percentile(accs, lo))
    upper = float(np.percentile(accs, hi))
    return acc_mean, lower, upper


@dataclass
class EvaluationResult:
    dataset: str
    method: str  # 'knn5' or 'linear'
    accuracy: float
    ci_lower: float
    ci_upper: float
    feature_dim: int
    best_c: float | None
    n_train: int
    n_val: int
    n_test: int
    seed: int
    model: str
    name: str

    def to_row(self) -> dict:
        return self.__dict__.copy()


def embed_split(
    model: BenchModel, dataloader, device: torch.device, verbose: bool
) -> tuple[np.ndarray, np.ndarray]:
    # Leverage existing util which handles different model output shapes.
    return extract_features(model, dataloader, device, transforms=None, verbose=verbose)


def evaluate_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_bootstrap: int,
    verbose: bool = False,
) -> tuple[float, float, float]:
    if verbose:
        logger.info(
            f"[KNN] Fit KNN5 (train={len(x_train)}, test={len(x_test)}, boot={n_bootstrap})"
        )
    clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    acc_mean, lo, hi = bootstrap_accuracy(y_test, preds, n_boot=n_bootstrap, seed=seed)
    if verbose:
        logger.info(f"[KNN] Test accuracy={acc_mean:.4f} (CI {lo:.4f}-{hi:.4f})")
    return acc_mean, lo, hi


def evaluate_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_values: Sequence[float],
    seed: int,
    n_bootstrap: int,
    merge_val: bool,
    device: str,
    verbose: bool = False,
) -> tuple[float, float, float, float]:
    """Sweep C values using validation set, then retrain and evaluate.

    Notes:
        - Standardization was removed from the core LogisticRegression; we keep
          embeddings raw here for consistency.
        - Inputs are numpy arrays; they are converted to torch tensors once.
        - Validation sweep trains separate lightweight models for each C.
        - Final model is retrained (optionally on train+val) with higher max_iter.
    """
    best_c: float | None = None
    best_val_acc = -1.0

    # Convert once to tensors (int64 labels for classification)
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train).long()
    x_val_tensor = torch.from_numpy(x_val)
    y_val_tensor = torch.from_numpy(y_val).long()
    x_test_tensor = torch.from_numpy(x_test)
    y_test_tensor = torch.from_numpy(y_test).long()

    if verbose:
        logger.info(
            f"[LogReg] C sweep start over {len(c_values)} values (train={len(x_train)}, val={len(x_val)})"
        )
        c_value_iterator = tqdm(c_values, desc="C values", leave=False)
    else:
        c_value_iterator = c_values

    for idx, c in enumerate(c_value_iterator):
        model = LogisticRegression(
            C=c,
            max_iter=2000,
            tol=1e-6,
            random_state=seed,
            device=device,
        )
        model.fit(x_train_tensor, y_train_tensor)
        val_pred = model.predict(x_val_tensor)
        acc_val = accuracy_score(y_val, val_pred)
        if verbose and (idx < 10 or idx % 50 == 0):
            logger.info(f"[LogReg] C={c:.4g} val_acc={acc_val:.4f}")
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            best_c = c

    assert best_c is not None, "C sweep failed to select a value"
    if verbose:
        logger.info(f"[LogReg] Best C={best_c:.4g} val_acc={best_val_acc:.4f}")

    # Prepare final training tensors
    if merge_val:
        x_final_np = np.concatenate([x_train, x_val], axis=0)
        y_final_np = np.concatenate([y_train, y_val], axis=0)
        x_final = torch.from_numpy(x_final_np)
        y_final = torch.from_numpy(y_final_np).long()
    else:
        x_final = x_train_tensor
        y_final = y_train_tensor

    final_model = LogisticRegression(
        C=best_c,
        max_iter=4000,
        tol=1e-6,
        random_state=seed,
        device=device,
    )
    final_model.fit(x_final, y_final)
    test_preds = final_model.predict(x_test_tensor)

    acc, lo, hi = bootstrap_accuracy(y_test, test_preds, n_boot=n_bootstrap, seed=seed)
    if verbose:
        logger.info(
            f"[LogReg] Test accuracy={acc:.4f} (CI {lo:.4f}-{hi:.4f}) using C={best_c:.4g}; train_final={len(x_final)} test={len(x_test)}"
        )
    return acc, lo, hi, float(best_c)

# (logging already imported above)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: D401
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset_names = _expand_dataset_list(cfg.dataset.names)
    device = torch.device(cfg.device)

    # Output file path
    output_path = cfg.output

    def _append_rows_atomic(path: str, rows: list[dict]) -> None:
        """Append rows to CSV atomically with advisory file lock.

        Ensures that if multiple processes start roughly simultaneously, they
        won't overwrite each other's output. Creates file if absent, writes header
        only if file was empty prior to append.
        """
        if not rows:
            return
        df_local = pd.DataFrame(rows)
        # Open file in append+read mode; create if not exists
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        with os.fdopen(fd, "r+", closefd=True) as f:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            empty = f.tell() == 0
            # Prepare CSV in memory
            buf = io.StringIO()
            df_local.to_csv(buf, header=empty, index=False)
            f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    all_rows: list[dict] = []
    c_start, c_stop, c_num = cfg.eval.c_range
    c_values = 10 ** np.linspace(float(c_start), float(c_stop), int(c_num))
    c_values_list = [float(v) for v in c_values.tolist()]
    
    # Load existing results if resume mode is enabled
    completed_runs: set[tuple[str, str, str, str]] = set()
    if cfg.resume and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(cfg.output)
            # Track (dataset, method, model) tuples that are already computed
            for _, row in existing_df.iterrows():
                completed_runs.add((
                    str(row.get("dataset", "")),
                    str(row.get("method", "")),
                    str(row.get("model", "")),
                    str(row.get("name", ""))
                ))
            print(f"Resume mode: Found {len(completed_runs)} existing results in {cfg.output}")
            print(f"Will skip already-computed (dataset, method, model) combinations.")
        except Exception as e:
            print(f"Warning: Could not load existing results for resume: {e}")
            completed_runs = set()

    for ds_name in tqdm(dataset_names, desc="Datasets"):
        # Check if we can skip this dataset entirely
        knn_key = (ds_name, "knn5", cfg.model._target_, cfg.model.name)
        linear_key = (ds_name, "linear", cfg.model._target_, cfg.model.name)
        skip_knn = cfg.resume and knn_key in completed_runs
        skip_linear = cfg.resume and linear_key in completed_runs
        skip_linear = skip_linear or getattr(cfg.eval, "skip_linear", False)
        
        if skip_knn and skip_linear:
            if cfg.verbose:
                print(f"[{ds_name}] Skipping entirely (all methods already computed)")
            continue
        
        result = get_datasets(
            dataset_name=ds_name,
            partition_name=cfg.dataset.partition,
            batch_size=cfg.dataset.batch_size,
            normalization=cfg.dataset.normalization,
            return_val=True,
            image_size=getattr(cfg.dataset, "image_size", None),
            interpolation=getattr(cfg.dataset, "interpolation", "bicubic"),
        )
        if result is None or not isinstance(result, tuple) or len(result) != 4:
            print(f"Skipping dataset {ds_name} (unexpected return)")
            continue
        train_dataset, train_loader, val_loader, test_loader = result

        # Determine channels & instantiate (per dataset re-init). We avoid inserting
        # raw dataset objects into the OmegaConf tree (unsupported). Instead we
        # conditionally supply the dataset only for empirical RCF models.
        # Access first sample to infer channel count (dataset returns mapping-like Sample)
        first_sample = train_dataset[0]  # type: ignore[index]
        num_channels = first_sample["image"].shape[0]  # type: ignore[index]
        model_cfg = OmegaConf.merge(cfg.model, {"num_channels": num_channels})

        needs_dataset = (
            hasattr(cfg.model, "mode")
            and str(cfg.model._target_).endswith("RCFBench")
            and str(cfg.model.mode) == "empirical"
        )
        if needs_dataset:
            target_path: str = cfg.model._target_
            module_name, class_name = target_path.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            ModelClass = getattr(module, class_name)
            model: BenchModel = ModelClass(
                num_channels=num_channels,
                features=cfg.model.features,
                kernel_size=cfg.model.kernel_size,
                mode=cfg.model.mode,
                stats_mode=cfg.model.stats_mode,
                seed=getattr(cfg.model, "seed", None),
                dataset=train_dataset,
            )
        else:
            model: BenchModel = instantiate(model_cfg)  # type: ignore
        model.to(device)
        model.eval()

        if cfg.verbose:
            print(f"[{ds_name}] Embedding train set ({len(train_loader)} batches)...")
        x_train, y_train = embed_split(model, train_loader, device, verbose=cfg.verbose)
        if cfg.verbose:
            print(f"[{ds_name}] Embedding val set ({len(val_loader)} batches)...")
        x_val, y_val = embed_split(model, val_loader, device, verbose=cfg.verbose)
        if cfg.verbose:
            print(f"[{ds_name}] Embedding test set ({len(test_loader)} batches)...")
        x_test, y_test = embed_split(model, test_loader, device, verbose=cfg.verbose)
        feature_dim = x_train.shape[1]

        if not skip_knn:
            knn_acc, knn_lo, knn_hi = evaluate_knn(
                x_train, y_train, x_test, y_test, cfg.seed, cfg.eval.bootstrap, verbose=cfg.verbose
            )
            all_rows.append(
                EvaluationResult(
                    dataset=ds_name,
                    method="knn5",
                    accuracy=knn_acc,
                    ci_lower=knn_lo,
                    ci_upper=knn_hi,
                    feature_dim=feature_dim,
                    best_c=None,
                    n_train=len(x_train),
                    n_val=len(x_val),
                    n_test=len(x_test),
                    seed=cfg.seed,
                    model=cfg.model._target_,
                    name=cfg.model.name,
                ).to_row()
            )

        if not skip_linear:
            if cfg.verbose:
                print(
                    f"[{ds_name}] Running LogisticRegression sweep over {len(c_values_list)} C values..."
                )
            lin_acc, lin_lo, lin_hi, best_c = evaluate_logistic(
                x_train,
                y_train,
                x_val,
                y_val,
                x_test,
                y_test,
                c_values=c_values_list,
                seed=cfg.seed,
                n_bootstrap=cfg.eval.bootstrap,
                merge_val=cfg.eval.merge_val,
                device=cfg.device,
                verbose=cfg.verbose,
            )
            all_rows.append(
                EvaluationResult(
                    dataset=ds_name,
                    method="linear",
                    accuracy=lin_acc,
                    ci_lower=lin_lo,
                    ci_upper=lin_hi,
                    feature_dim=feature_dim,
                    best_c=best_c,
                    n_train=len(x_train),
                    n_val=len(x_val),
                    n_test=len(x_test),
                    seed=cfg.seed,
                    model=cfg.model._target_,
                    name=cfg.model.name,
                ).to_row()
            )

        _append_rows_atomic(output_path, all_rows)
        all_rows.clear()

    print(f"Benchmark complete. Results appended to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    # Hydra provides cfg automatically; this call signature is correct.
    main()  # type: ignore[misc]
