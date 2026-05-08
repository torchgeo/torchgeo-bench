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
from torch.utils.data import ConcatDataset, DataLoader
from torchgeo.datasets.errors import DatasetNotFoundError
from tqdm import tqdm

from torchgeo_bench.datasets import (
    get_bench_dataset_class,
    get_datasets,
    list_datasets,
)
from torchgeo_bench.intrinsic_dim import compute_intrinsic_dim
from torchgeo_bench.knn import KNNClassifier
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
)
from torchgeo_bench.segmentation_task import SegmentationSolver, SegMetrics
from torchgeo_bench.segmentation_viz import save_segmentation_viz
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
            return list_datasets()
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)


def _normalize_bands_value(bands: object) -> str:
    """Canonicalize the ``cfg.dataset.bands`` value for logging/CSV/resume.

    Hydra hands us either ``"rgb"``/``"all"``, an explicit list (``ListConfig``
    or ``list[str]``), or ``None``.  Reduce all of those to a stable string so
    that the resume key and the CSV column are comparable across runs.

    Args:
        bands: The raw ``cfg.dataset.bands`` value.

    Returns:
        A stable string representation: ``"rgb"``, ``"all"``, or a
        comma-joined explicit band list (e.g. ``"red,green,blue,nir"``).
    """
    if bands is None:
        return "all"
    if isinstance(bands, str):
        return bands
    try:
        items = [str(b) for b in bands]
    except TypeError:
        return str(bands)
    return ",".join(items)


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
    method: str  # 'knn5', 'linear', or seg head type
    metric_name: str  # 'accuracy', 'micro_mAP', or 'mIoU' (primary metric)
    metric_value: float
    ci_lower: float
    ci_upper: float
    feature_dim: int
    best_c: float | None
    best_lr: float | None
    best_batch_size: int | None
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
    bands: str
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int
    # Segmentation-only metrics (None for classification rows)
    fw_iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None

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
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Evaluate KNN classifier. Auto-detects single-label vs multi-label from y shape."""
    multi_label = y_train.ndim == 2
    clf = KNNClassifier(n_neighbors=5)
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


def _make_seg_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_loader: DataLoader,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and train+val DataLoaders for a given batch size.

    Args:
        train_dataset: Training split dataset.
        val_dataset: Validation split dataset.
        test_loader: Pre-built test loader (reused as-is).
        batch_size: Batch size for the new loaders.

    Returns:
        Tuple of (train_loader, val_loader, train_val_loader).
    """
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": test_loader.num_workers,
        "pin_memory": test_loader.pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    train_val_loader = DataLoader(
        ConcatDataset([train_dataset, val_dataset]), shuffle=True, **loader_kwargs
    )
    return train_loader, val_loader, train_val_loader


def _build_seg_probe_and_solver(
    model: torch.nn.Module,
    num_classes: int,
    eval_cfg: DictConfig,
    device: torch.device,
    lr: float,
) -> tuple[SegmentationProbe, SegmentationSolver]:
    """Instantiate a fresh SegmentationProbe and SegmentationSolver.

    Args:
        model: Frozen backbone (shared across calls; only the head is re-created).
        num_classes: Number of segmentation classes.
        eval_cfg: Merged evaluation config with segmentation sub-config.
        device: Target device.
        lr: Learning rate for the solver optimizer.

    Returns:
        Tuple of (probe, solver).
    """
    probe = SegmentationProbe(
        backbone=model,
        layer_names=eval_cfg.segmentation.layers,
        num_classes=num_classes,
        head_type=eval_cfg.segmentation.head_type,
        freeze_backbone=True,
    )
    criterion = instantiate(eval_cfg.segmentation.criterion)
    solver = SegmentationSolver(
        model=probe,
        num_classes=num_classes,
        lr=lr,
        device=str(device),
        criterion=criterion,
        lr_scheduler=eval_cfg.segmentation.get("lr_scheduler", "cosine"),
        ignore_index=eval_cfg.segmentation.get("ignore_index", 255),
    )
    return probe, solver


def evaluate_intrinsic_dim(
    splits: dict[str, np.ndarray],
    estimators: Sequence[str],
    selected_splits: Sequence[str],
    device: str | None,
    max_samples: int | None,
    seed: int,
    common_meta: dict,
    feature_dim: int,
    n_counts: dict[str, int],
    verbose: bool = False,
) -> list[dict]:
    """Compute intrinsic-dimension metrics over selected splits and return CSV rows.

    Each (split, estimator) yields one row with ``method="intrinsic_dim"`` and
    ``metric_name=f"id_{estimator}_{split}"``.
    """
    rows: list[dict] = []
    for split_name in selected_splits:
        if split_name not in splits:
            logger.warning(f"[intrinsic-dim] unknown split '{split_name}', skipping")
            continue
        X = splits[split_name]
        if verbose:
            logger.info(
                f"[intrinsic-dim] split={split_name} X{X.shape} "
                f"estimators={list(estimators)} device={device}"
            )
        dims = compute_intrinsic_dim(
            X,
            estimators=list(estimators),
            device=device,
            max_samples=max_samples,
            seed=seed,
        )
        for est_name, dim in dims.items():
            rows.append(
                EvaluationResult(
                    **common_meta,
                    method="intrinsic_dim",
                    metric_name=f"id_{est_name}_{split_name}",
                    metric_value=float(dim),
                    ci_lower=0.0,
                    ci_upper=0.0,
                    feature_dim=feature_dim,
                    best_c=None,
                    best_lr=None,
                    best_batch_size=None,
                    n_train=n_counts.get("train", 0),
                    n_val=n_counts.get("val", 0),
                    n_test=n_counts.get("test", 0),
                ).to_row()
            )
    return rows


def evaluate_segmentation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DictConfig,
    num_classes: int,
    device: torch.device,
    collect_preds: bool = False,
) -> "tuple[SegMetrics, int, float | None, int | None, torch.Tensor | None]":
    """Evaluate segmentation performance using a frozen-backbone segmentation probe.

    Trains a lightweight segmentation head on top of the frozen backbone and
    evaluates mIoU on the test split. Optionally pre-caches backbone features for
    faster training across epochs.

    Args:
        model: Frozen backbone model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        cfg: Full Hydra config.
        num_classes: Number of segmentation classes.
        device: Torch device.
        collect_preds: If True, collect and return test predictions as (N, H, W) tensor.

    Returns:
        Tuple of (metrics_dict, feature_dim, None, None, preds_or_None).
        ``preds_or_None`` is None when collect_preds is False.
    """
    # Merge model-specific eval config if present
    eval_cfg = cfg.eval
    if "eval" in cfg.model and cfg.model.eval is not None:
        eval_cfg = OmegaConf.merge(eval_cfg, cfg.model.eval)
    if "segmentation" not in eval_cfg:
        raise ValueError("Segmentation evaluation config missing for the model.")

    seg_cfg = eval_cfg.segmentation
    epochs = seg_cfg.epochs
    use_cache = seg_cfg.get("cache_features", True)
    cache_dtype_str = seg_cfg.get("cache_dtype", "float16")
    cache_dtype = torch.float16 if cache_dtype_str == "float16" else torch.float32

    probe, solver = _build_seg_probe_and_solver(model, num_classes, eval_cfg, device, seg_cfg.lr)
    if use_cache and probe.freeze_backbone:
        logger.info("Caching backbone features for train and val splits...")
        train_cache = probe.extract_segmentation_features(train_loader, cache_dtype=cache_dtype)
        val_cache = probe.extract_segmentation_features(val_loader, cache_dtype=cache_dtype)
        test_cache = probe.extract_segmentation_features(test_loader, cache_dtype=cache_dtype)
        solver.fit_cached(
            train_cache=train_cache,
            val_cache=val_cache,
            batch_size=seg_cfg.get("batch_size", 64),
            epochs=epochs,
            verbose=cfg.verbose,
        )
        eval_result = solver.evaluate_cached(
            test_cache,
            batch_size=seg_cfg.get("batch_size", 64),
            collect_preds=collect_preds,
        )
    else:
        solver.fit(
            train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=cfg.verbose
        )
        eval_result = solver.evaluate(test_loader, collect_preds=collect_preds)

    if collect_preds:
        metrics, preds = eval_result
    else:
        metrics, preds = eval_result, None
    return metrics, sum(probe.channels_list), None, None, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def append_rows_atomic(path: str, rows: list[dict]) -> None:
    """Append rows to a CSV atomically, with advisory file lock and schema healing.

    Behavior:

    - Empty/missing file: writes the header derived from ``rows`` and the rows.
    - Existing file whose header matches ``rows[0]`` keys exactly: appends
      rows without rewriting the header (fast path).
    - Existing file with a different schema (e.g. ``EvaluationResult`` gained
      a field since the file was first written): the file is rewritten with
      the unioned schema so every value lives under a named column instead
      of being silently stuffed into an unnamed position.

    Args:
        path: Output CSV path; created if missing.
        rows: List of dicts to append.  All dicts should share the same keys.
    """
    if not rows:
        return
    df_local = pd.DataFrame(rows)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    with os.fdopen(fd, "r+", closefd=True) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            empty = f.tell() == 0
            buf = io.StringIO()
            if empty:
                df_local.to_csv(buf, header=True, index=False)
                f.write(buf.getvalue())
            else:
                f.seek(0)
                existing_df = pd.read_csv(f)
                if list(existing_df.columns) == list(df_local.columns):
                    df_local.to_csv(buf, header=False, index=False)
                    f.seek(0, os.SEEK_END)
                    f.write(buf.getvalue())
                else:
                    extra = [c for c in existing_df.columns if c not in df_local.columns]
                    ordered = list(df_local.columns) + extra
                    combined = pd.concat(
                        [existing_df, df_local], ignore_index=True, sort=False
                    ).reindex(columns=ordered)
                    logger.warning(
                        "CSV schema drift detected at %s: existing columns %s, "
                        "new columns %s. Rewriting with unioned schema %s.",
                        path,
                        list(existing_df.columns),
                        list(df_local.columns),
                        ordered,
                    )
                    f.seek(0)
                    f.truncate()
                    combined.to_csv(buf, header=True, index=False)
                    f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


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

    all_rows: list[dict] = []
    c_start, c_stop, c_num = cfg.eval.c_range
    c_values = 10 ** np.linspace(float(c_start), float(c_stop), int(c_num))
    c_values_list = [float(v) for v in c_values.tolist()]

    # Load existing results if resume mode is enabled
    completed_runs: set[tuple[str, ...]] = set()
    if cfg.resume and os.path.exists(output_path):
        existing_df = pd.read_csv(cfg.output)
        # Track (dataset, method, model, name, normalization, image_size,
        # interpolation, partition, bands) tuples
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
                    str(row.get("bands", "")),
                )
            )
        logger.info(f"Resume mode: Found {len(completed_runs)} existing results in {cfg.output}")
        logger.info("Will skip already-computed (dataset, method, model, config) combinations.")

    # Datasets always emit raw values; the model owns normalization.  The
    # CSV column is kept for back-compat but pinned to a literal so old/new
    # rows are clearly not comparable across the model-normalization refactor.
    normalization = "raw"
    bands_value = _normalize_bands_value(getattr(cfg.dataset, "bands", "rgb"))

    for ds_name in tqdm(dataset_names, desc="Datasets"):
        # Resolve metadata via the BenchDataset registry (no I/O).
        try:
            ds_cls = get_bench_dataset_class(ds_name)
        except KeyError:
            logger.warning(f"Skipping dataset {ds_name} (not in registry)")
            continue

        # Check if we can skip this dataset entirely
        # Include dataset config params to ensure we only skip with matching settings
        config_tuple = (
            normalization,
            str(getattr(cfg.dataset, "image_size", None)),
            getattr(cfg.dataset, "interpolation", "bicubic"),
            cfg.dataset.partition,
            bands_value,
        )

        # Merge model-specific eval config early so resume key and result rows
        # reflect the actual head_type used, not the global default.
        eval_cfg_merged = OmegaConf.merge(
            cfg.eval,
            cfg.model.eval if "eval" in cfg.model and cfg.model.eval is not None else {},
        )

        # Check resume for standard methods
        knn_key = (ds_name, "knn5", cfg.model._target_, cfg.model.name, *config_tuple)
        linear_key = (ds_name, "linear", cfg.model._target_, cfg.model.name, *config_tuple)

        seg_method = f"seg-{eval_cfg_merged.segmentation.head_type}"
        seg_key = (ds_name, seg_method, cfg.model._target_, cfg.model.name, *config_tuple)
        id_key = (ds_name, "intrinsic_dim", cfg.model._target_, cfg.model.name, *config_tuple)

        try:
            result = get_datasets(
                dataset_name=ds_name,
                partition_name=cfg.dataset.partition,
                batch_size=cfg.dataset.batch_size,
                return_val=True,
                image_size=getattr(cfg.dataset, "image_size", None),
                interpolation=getattr(cfg.dataset, "interpolation", "bicubic"),
                bands=getattr(cfg.dataset, "bands", "rgb"),
            )
        except (FileNotFoundError, DatasetNotFoundError) as exc:
            logger.warning(f"Skipping dataset {ds_name} (data not found: {exc})")
            continue
        if result is None or not isinstance(result, tuple) or len(result) != 4:
            logger.warning(f"Skipping dataset {ds_name} (unexpected return)")
            continue
        train_dataset, train_loader, val_loader, test_loader = result

        # Use metadata from the BenchDataset class
        num_channels = train_dataset[0]["image"].shape[0]
        is_segmentation = ds_cls.task == "segmentation"
        is_multilabel = ds_cls.multilabel
        num_classes = ds_cls.num_classes

        # Build the BandSpec list that matches the actual loaded channels.
        bench_for_bands = ds_cls()
        bands_resolved = (
            tuple(bench_for_bands.rgb_bands)
            if cfg.dataset.bands == "rgb"
            else None
            if cfg.dataset.bands in ("all", None)
            else tuple(cfg.dataset.bands)
        )
        bands_list = bench_for_bands.select_band_specs(bands_resolved)
        assert len(bands_list) == num_channels, (
            f"BandSpec count {len(bands_list)} != tensor channel count {num_channels} "
            f"for dataset {ds_name}; sample-level canonicalization may have changed shape."
        )

        # Resume check for segmentation
        if is_segmentation and cfg.resume and seg_key in completed_runs:
            if cfg.verbose:
                logger.info(f"[{ds_name}] Skipping segmentation (already computed)")
            continue

        # Instantiate Backbone — pass `bands` post-hoc so Hydra never tries
        # to OmegaConf-ify the BandSpec list.  `_convert_="object"` keeps
        # the rest of the model config as plain Python primitives.
        is_rcf_empirical = (
            hasattr(cfg.model, "mode")
            and str(cfg.model._target_).endswith("RCFBench")
            and str(cfg.model.mode) == "empirical"
        )
        instantiate_kwargs: dict = {"bands": bands_list, "_convert_": "object"}
        if is_rcf_empirical:
            instantiate_kwargs["dataset"] = train_dataset
        model: BenchModel = instantiate(cfg.model, **instantiate_kwargs)
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
            "bands": bands_value,
            "c_range_start": c_start,
            "c_range_stop": c_stop,
            "c_range_num": c_num,
            "merge_val": cfg.eval.merge_val,
            "bootstrap": cfg.eval.bootstrap,
        }

        if is_segmentation:
            seg_cfg_merged = OmegaConf.merge(
                cfg.eval,
                cfg.model.eval if "eval" in cfg.model and cfg.model.eval is not None else {},
            ).segmentation
            save_viz = seg_cfg_merged.get("save_viz", False)
            metrics, feat_dim, best_lr, best_bs, preds = evaluate_segmentation(
                model,
                train_loader,
                val_loader,
                test_loader,
                cfg,
                num_classes,
                device,
                collect_preds=save_viz,
            )
            all_rows.append(
                EvaluationResult(
                    **common_meta,
                    method=seg_method,
                    metric_name="mIoU",
                    metric_value=metrics.get("mIoU", float("nan")),
                    ci_lower=0.0,
                    ci_upper=0.0,
                    feature_dim=feat_dim,
                    best_c=None,
                    best_lr=best_lr,
                    best_batch_size=best_bs,
                    n_train=len(train_dataset),
                    n_val=len(val_loader.dataset),
                    n_test=len(test_loader.dataset),
                    fw_iou=metrics.get("fw_IoU"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1=metrics.get("f1"),
                ).to_row()
            )
            if save_viz and preds is not None:
                rgb_indices = ds_cls().rgb_indices or [0, 1, 2]
                # Collect images and GT masks from test_loader (cheap pass, no backbone)
                test_imgs, test_gts = [], []
                for _batch in test_loader:
                    if isinstance(_batch, dict):
                        test_imgs.append(_batch["image"])
                        _m = _batch["mask"]
                    else:
                        test_imgs.append(_batch[0])
                        _m = _batch[1]
                    if _m.ndim == 4:
                        _m = _m.squeeze(1)
                    test_gts.append(_m.long())
                test_imgs_t = torch.cat(test_imgs, dim=0)
                test_gts_t = torch.cat(test_gts, dim=0)
                ignore_idx = seg_cfg_merged.get("ignore_index", 255)
                n_viz = seg_cfg_merged.get("n_viz_samples", 8)
                viz_dir = seg_cfg_merged.get("viz_dir", "viz")
                _class_names = list(getattr(train_dataset, "classes", None) or []) or None
                save_segmentation_viz(
                    out_dir=viz_dir,
                    model_name=cfg.model.name,
                    dataset_name=ds_name,
                    images=test_imgs_t,
                    gt_masks=test_gts_t,
                    pred_masks=preds,
                    num_classes=num_classes,
                    rgb_indices=rgb_indices,
                    ignore_index=ignore_idx,
                    n_samples=n_viz,
                    class_names=_class_names,
                )
        else:
            # Classification (single-label or multi-label)
            metric_name = "micro_mAP" if is_multilabel else "accuracy"

            skip_knn = cfg.resume and knn_key in completed_runs
            skip_linear = (cfg.resume and linear_key in completed_runs) or getattr(
                cfg.eval, "skip_linear", False
            )
            id_cfg = getattr(cfg.eval, "intrinsic_dim", None)
            id_enabled = bool(id_cfg and id_cfg.get("enabled", False))
            skip_id = (not id_enabled) or (cfg.resume and id_key in completed_runs)

            if skip_knn and skip_linear and skip_id:
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
                        best_lr=None,
                        best_batch_size=None,
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
                        best_lr=None,
                        best_batch_size=None,
                        n_train=len(x_train),
                        n_val=len(x_val),
                        n_test=len(x_test),
                    ).to_row()
                )

            if not skip_id:
                id_rows = evaluate_intrinsic_dim(
                    splits={"train": x_train, "val": x_val, "test": x_test},
                    estimators=list(id_cfg.estimators),
                    selected_splits=list(id_cfg.splits),
                    device=id_cfg.get("device", None) or cfg.device,
                    max_samples=id_cfg.get("max_samples", None),
                    seed=cfg.seed,
                    common_meta=common_meta,
                    feature_dim=feature_dim,
                    n_counts={
                        "train": len(x_train),
                        "val": len(x_val),
                        "test": len(x_test),
                    },
                    verbose=cfg.verbose,
                )
                all_rows.extend(id_rows)

        append_rows_atomic(output_path, all_rows)
        all_rows.clear()

    logger.info(f"Benchmark complete. Results appended to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    # Hydra provides cfg automatically; this call signature is correct.
    main()  # type: ignore[misc]
