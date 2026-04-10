"""Benchmark script for torchgeo-bench."""

import fcntl
import io
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchgeo_bench.dataset_info import list_available_datasets, load_dataset_info
from torchgeo_bench.datasets import get_datasets, is_dataset_available
from torchgeo_bench.knn import KNNClassifier
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.segmentation_probe import SegmentationProbe
from torchgeo_bench.segmentation_task import SegmentationSolver
from torchgeo_bench.utils import extract_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _expand_dataset_list(names: str | Sequence[str]) -> list[str]:
    """Expand dataset names to a flat list.

    Args:
        names: Dataset name(s) — ``"all"``, comma-separated string, or sequence.

    Returns:
        List of individual dataset name strings.
    """
    if isinstance(names, str):
        if names == "all":
            return list_available_datasets()
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Compute accuracy with bootstrapped confidence interval.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        n_boot: Number of bootstrap resamples.
        ci: Confidence interval width in percent.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (mean_accuracy, ci_lower, ci_upper).
    """
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


def bootstrap_map(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap micro-averaged mean Average Precision."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    map_mean = float(average_precision_score(y_true, y_scores, average="micro"))
    valid_maps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        # Skip degenerate resamples with no positive labels
        if yt.sum() == 0:
            continue
        valid_maps.append(average_precision_score(yt, y_scores[idx], average="micro"))
    if not valid_maps:
        return map_mean, map_mean, map_mean
    maps = np.array(valid_maps, dtype=np.float32)
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = float(np.percentile(maps, lo))
    upper = float(np.percentile(maps, hi))
    return map_mean, lower, upper


@dataclass
class EvaluationResult:
    """Container for a single evaluation result row."""

    dataset: str
    method: str  # 'knn5' or 'linear' seg_linear, seg_conv
    metric_name: str  # 'accuracy' or 'mIoU'
    metric_value: float
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
    normalization: str
    image_size: int | None
    interpolation: str
    partition: str
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int

    def to_row(self) -> dict:
        """Convert to a flat dictionary suitable for CSV/DataFrame export."""
        return self.__dict__.copy()


def embed_split(
    model: BenchModel, dataloader: DataLoader, device: torch.device, verbose: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a data split.

    Args:
        model: The benchmark model to extract features with.
        dataloader: DataLoader for the split.
        device: Torch device to run inference on.
        verbose: Whether to show a progress bar.

    Returns:
        Tuple of (features, labels) as NumPy arrays.
    """
    return extract_features(model, dataloader, device, transforms=None, verbose=verbose)


def evaluate_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_bootstrap: int,
    device: str = "cpu",
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Evaluate KNN classifier. Auto-detects single-label vs multi-label from y shape."""
    multi_label = y_train.ndim == 2
    clf = KNNClassifier(n_neighbors=5, device=device)
    clf.fit(x_train, y_train)

    if multi_label:
        if verbose:
            logger.info(f"[KNN] Fit KNN5 multilabel (train={len(x_train)}, test={len(x_test)})")
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_map(y_test, y_scores, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info(f"[KNN] Test micro_mAP={metric:.4f} (CI {lo:.4f}-{hi:.4f})")
    else:
        if verbose:
            logger.info(
                f"[KNN] Fit KNN5 (train={len(x_train)}, test={len(x_test)}, boot={n_bootstrap})"
            )
        preds = clf.predict(x_test)
        metric, lo, hi = bootstrap_accuracy(y_test, preds, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info(f"[KNN] Test accuracy={metric:.4f} (CI {lo:.4f}-{hi:.4f})")

    return metric, lo, hi


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
    """Sweep C values, retrain, and evaluate. Auto-detects single/multi-label from y shape."""
    multi_label = y_train.ndim == 2
    best_c: float | None = None
    best_val_score = -1.0

    x_train_tensor = torch.from_numpy(x_train)
    x_val_tensor = torch.from_numpy(x_val)
    x_test_tensor = torch.from_numpy(x_test)

    if multi_label:
        y_train_tensor = torch.from_numpy(y_train).float()
        label_tag = "LogReg-ML"
    else:
        y_train_tensor = torch.from_numpy(y_train).long()
        label_tag = "LogReg"

    if verbose:
        logger.info(
            f"[{label_tag}] C sweep start over {len(c_values)} values "
            f"(train={len(x_train)}, val={len(x_val)})"
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
            multi_label=multi_label,
        )
        model.fit(x_train_tensor, y_train_tensor)

        if multi_label:
            val_scores = model.predict_proba(x_val_tensor)
            val_metric = float(average_precision_score(y_val, val_scores, average="micro"))
        else:
            val_pred = model.predict(x_val_tensor)
            val_metric = accuracy_score(y_val, val_pred)

        if verbose and (idx < 10 or idx % 50 == 0):
            logger.info(f"[{label_tag}] C={c:.4g} val_score={val_metric:.4f}")
        if val_metric > best_val_score:
            best_val_score = val_metric
            best_c = c

    assert best_c is not None, "C sweep failed to select a value"
    if verbose:
        logger.info(f"[{label_tag}] Best C={best_c:.4g} val_score={best_val_score:.4f}")

    # Prepare final training tensors
    if merge_val:
        x_final_np = np.concatenate([x_train, x_val], axis=0)
        y_final_np = np.concatenate([y_train, y_val], axis=0)
        x_final = torch.from_numpy(x_final_np)
        y_final = (
            torch.from_numpy(y_final_np).float()
            if multi_label
            else torch.from_numpy(y_final_np).long()
        )
    else:
        x_final = x_train_tensor
        y_final = y_train_tensor

    final_model = LogisticRegression(
        C=best_c,
        max_iter=4000,
        tol=1e-6,
        random_state=seed,
        device=device,
        multi_label=multi_label,
    )
    final_model.fit(x_final, y_final)

    if multi_label:
        test_scores = final_model.predict_proba(x_test_tensor)
        metric, lo, hi = bootstrap_map(y_test, test_scores, n_boot=n_bootstrap, seed=seed)
    else:
        test_preds = final_model.predict(x_test_tensor)
        metric, lo, hi = bootstrap_accuracy(y_test, test_preds, n_boot=n_bootstrap, seed=seed)

    if verbose:
        logger.info(
            f"[{label_tag}] Test score={metric:.4f} (CI {lo:.4f}-{hi:.4f}) "
            f"using C={best_c:.4g}; train_final={len(x_final)} test={len(x_test)}"
        )
    return metric, lo, hi, float(best_c)


def evaluate_segmentation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DictConfig,
    num_classes: int,
    device: torch.device,
) -> tuple[float, int]:
    """Evaluate segmentation performance using a segmentation probe and solver."""
    # merge with model specific eval config if present
    eval_cfg = cfg.eval
    if "eval" in cfg.model and cfg.model.eval is not None:
        eval_cfg = OmegaConf.merge(eval_cfg, cfg.model.eval)
    if "segmentation" not in eval_cfg:
        raise ValueError("Segmentation evaluation config missing for the model.")

    probe = SegmentationProbe(
        backbone=model,
        layer_names=eval_cfg.segmentation.layers,
        num_classes=num_classes,
        head_type=eval_cfg.segmentation.head_type,
        freeze_backbone=True,
    )
    criterion = instantiate(eval_cfg.segmentation.criterion, num_classes=num_classes) if "criterion" in eval_cfg.segmentation else None

    solver = SegmentationSolver(
        model=probe,
        num_classes=num_classes,
        lr=eval_cfg.segmentation.lr,
        device=str(device),
        criterion=criterion,
        lr_scheduler=eval_cfg.segmentation.get("lr_scheduler", "cosine"),
    )

    solver.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=eval_cfg.segmentation.epochs,
        verbose=cfg.verbose,
    )

    miou = solver.evaluate(test_loader)
    feature_dim = sum(probe.channels_list)

    return miou, feature_dim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the benchmark pipeline for all configured datasets and models."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset_names = _expand_dataset_list(cfg.dataset.names)
    device = torch.device(cfg.device)

    # Output file path
    output_path = cfg.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    def _append_rows_atomic(path: str, rows: list[dict]) -> None:
        """Append rows to CSV atomically with advisory file lock."""
        if not rows:
            return
        df_local = pd.DataFrame(rows)
        # Open file in read-write mode; create if not exists
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
    completed_runs: set[tuple[str, str, str, str, str, str, str, str]] = set()
    if cfg.resume and os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(cfg.output)
            # Track (dataset, method, model, name, normalization, image_size, interpolation, partition) tuples
            for _, row in existing_df.iterrows():
                completed_runs.add(
                    (
                        str(row.get("dataset", "")),
                        str(row.get("method", "")),
                        str(row.get("model", "")),
                        str(row.get("name", "")),
                        str(row.get("normalization", "")),
                        str(row.get("image_size", "")),
                        str(row.get("interpolation", "")),
                        str(row.get("partition", "")),
                    )
                )
            logger.info(
                f"Resume mode: Found {len(completed_runs)} existing results in {cfg.output}"
            )
            logger.info("Will skip already-computed (dataset, method, model, config) combinations.")
        except Exception as e:
            logger.warning(f"Could not load existing results for resume: {e}")
            completed_runs = set()

    # Model can override dataset normalization (e.g., torchgeo models that
    # need specific preprocessing).  Fall back to dataset.normalization.
    normalization = getattr(cfg.model, "normalization", None) or cfg.dataset.normalization

    for ds_name in tqdm(dataset_names, desc="Datasets"):
        # Load dataset metadata from config
        try:
            ds_info = load_dataset_info(ds_name)
        except FileNotFoundError:
            logger.warning(f"Skipping dataset {ds_name} (no config file found)")
            continue

        if not is_dataset_available(
            ds_name,
            geobench_root=getattr(cfg.dataset, "geobench_root", None),
            geobench_v2_root=getattr(cfg.dataset, "geobench_v2_root", None),
        ):
            logger.warning(f"Skipping dataset {ds_name} (data not found on disk), looked in {getattr(cfg.dataset, 'geobench_root', None)} and {getattr(cfg.dataset, 'geobench_v2_root', None)}")
            continue

        # Check if we can skip this dataset entirely
        # Include dataset config params to ensure we only skip with matching settings
        config_tuple = (
            normalization,
            str(getattr(cfg.dataset, "image_size", None)),
            getattr(cfg.dataset, "interpolation", "bicubic"),
            cfg.dataset.partition,
        )

        # Check resume for standard methods
        knn_key = (ds_name, "knn5", cfg.model._target_, cfg.model.name, *config_tuple)
        linear_key = (ds_name, "linear", cfg.model._target_, cfg.model.name, *config_tuple)

        seg_method = f"seg-{cfg.eval.segmentation.head_type}"
        seg_key = (ds_name, seg_method, cfg.model._target_, cfg.model.name, *config_tuple)

        result = get_datasets(
            dataset_name=ds_name,
            partition_name=cfg.dataset.partition,
            batch_size=cfg.dataset.batch_size,
            normalization=normalization,
            return_val=True,
            image_size=getattr(cfg.dataset, "image_size", None),
            interpolation=getattr(cfg.dataset, "interpolation", "bicubic"),
            geobench_root=getattr(cfg.dataset, "geobench_root", None),
            geobench_v2_root=getattr(cfg.dataset, "geobench_v2_root", None),
            bands=getattr(cfg.dataset, "bands", "rgb"),
        )
        if result is None or not isinstance(result, tuple) or len(result) != 4:
            logger.warning(f"Skipping dataset {ds_name} (unexpected return)")
            continue
        train_dataset, train_loader, val_loader, test_loader = result

        # Use metadata from dataset config
        num_channels = train_dataset[0]["image"].shape[0]
        is_segmentation = ds_info.task == "segmentation"
        is_multilabel = ds_info.multilabel
        num_classes = ds_info.num_classes

        # Resume check for segmentation
        if is_segmentation and cfg.resume and seg_key in completed_runs:
            if cfg.verbose:
                logger.info(f"[{ds_name}] Skipping segmentation (already computed)")
            continue

        # Instantiate Backbone
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
            model = getattr(module, class_name)(
                num_channels=num_channels,
                features=cfg.model.features,
                kernel_size=cfg.model.kernel_size,
                mode=cfg.model.mode,
                stats_mode=cfg.model.stats_mode,
                seed=getattr(cfg.model, "seed", None),
                dataset=train_dataset,
            )
        else:
            model: BenchModel = instantiate(model_cfg)
        model.to(device).eval()

        # Shared Result metadata
        common_meta = {
            "dataset": ds_name,
            "seed": cfg.seed,
            "model": cfg.model._target_,
            "name": cfg.model.name,
            "normalization": normalization,
            "image_size": getattr(cfg.dataset, "image_size", None),
            "interpolation": getattr(cfg.dataset, "interpolation", "bicubic"),
            "partition": cfg.dataset.partition,
            "c_range_start": c_start,
            "c_range_stop": c_stop,
            "c_range_num": c_num,
            "merge_val": cfg.eval.merge_val,
            "bootstrap": cfg.eval.bootstrap,
        }

        if is_segmentation:
            miou, feat_dim = evaluate_segmentation(
                model, train_loader, val_loader, test_loader, cfg, num_classes, device
            )
            all_rows.append(
                EvaluationResult(
                    **common_meta,
                    method=cfg.eval.segmentation.head_type,
                    metric_name="mIoU",
                    metric_value=miou,
                    ci_lower=0.0,
                    ci_upper=0.0,
                    feature_dim=feat_dim,
                    best_c=None,
                    n_train=len(train_dataset),
                    n_val=len(val_loader.dataset),
                    n_test=len(test_loader.dataset),
                ).to_row()
            )
        else:
            # Classification (single-label or multi-label)
            metric_name = "micro_mAP" if is_multilabel else "accuracy"

            skip_knn = cfg.resume and knn_key in completed_runs
            skip_linear = (cfg.resume and linear_key in completed_runs) or getattr(
                cfg.eval, "skip_linear", False
            )

            if skip_knn and skip_linear:
                continue

            x_train, y_train = embed_split(model, train_loader, device, verbose=cfg.verbose)
            x_val, y_val = embed_split(model, val_loader, device, verbose=cfg.verbose)
            x_test, y_test = embed_split(model, test_loader, device, verbose=cfg.verbose)
            feature_dim = x_train.shape[1]

            if not skip_knn:
                knn_score, knn_lo, knn_hi = evaluate_knn(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    cfg.seed,
                    cfg.eval.bootstrap,
                    cfg.device,
                    verbose=cfg.verbose,
                )
                all_rows.append(
                    EvaluationResult(
                        **common_meta,
                        method="knn5",
                        metric_name=metric_name,
                        metric_value=knn_score,
                        ci_lower=knn_lo,
                        ci_upper=knn_hi,
                        feature_dim=feature_dim,
                        best_c=None,
                        n_train=len(x_train),
                        n_val=len(x_val),
                        n_test=len(x_test),
                    ).to_row()
                )

            if not skip_linear:
                lin_score, lin_lo, lin_hi, best_c = evaluate_logistic(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    c_values_list,
                    cfg.seed,
                    cfg.eval.bootstrap,
                    cfg.eval.merge_val,
                    cfg.device,
                    cfg.verbose,
                )
                all_rows.append(
                    EvaluationResult(
                        **common_meta,
                        method="linear",
                        metric_name=metric_name,
                        metric_value=lin_score,
                        ci_lower=lin_lo,
                        ci_upper=lin_hi,
                        feature_dim=feature_dim,
                        best_c=best_c,
                        n_train=len(x_train),
                        n_val=len(x_val),
                        n_test=len(x_test),
                    ).to_row()
                )

        _append_rows_atomic(output_path, all_rows)
        all_rows.clear()

    logger.info(f"Benchmark complete. Results appended to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    # Hydra provides cfg automatically; this call signature is correct.
    main()  # type: ignore[misc]
