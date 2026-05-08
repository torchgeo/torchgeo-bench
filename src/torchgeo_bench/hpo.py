"""Bayesian hyperparameter search for the segmentation probe (Optuna TPE)."""

import logging

import torch
from omegaconf import DictConfig

from torchgeo_bench.segmentation_probe import CachedFeaturesDataset, GPUTensorCache

logger = logging.getLogger(__name__)


def find_best_hparams(
    model: torch.nn.Module,
    train_cache: CachedFeaturesDataset,
    val_cache: CachedFeaturesDataset,
    num_classes: int,
    eval_cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run Optuna TPE to find the best LR and weight decay for the segmentation probe.

    Args:
        model: Frozen backbone (shared; only the head is rebuilt per trial).
        train_cache: Pre-extracted training features.
        val_cache: Pre-extracted validation features.
        num_classes: Number of segmentation classes.
        eval_cfg: Merged evaluation config (must contain an ``segmentation`` sub-config).
        device: Target device.

    Returns:
        Dict with ``"lr"`` and ``"weight_decay"`` keys set to the best found values.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError(
            "Optuna is required for HPO. Install it with: pip install torchgeo-bench[hpo]"
        ) from exc

    # Import here to avoid circular import at module level
    from torchgeo_bench.main import _build_seg_probe_and_solver

    seg_cfg = eval_cfg.segmentation
    n_trials: int = seg_cfg.get("n_trials", 10)
    hpo_epochs: int = seg_cfg.get("hpo_epochs", 5)
    lr_min: float = float(seg_cfg.get("lr_min", 1e-5))
    lr_max: float = float(seg_cfg.get("lr_max", 1e-2))
    wd_min: float = float(seg_cfg.get("wd_min", 1e-6))
    wd_max: float = float(seg_cfg.get("wd_max", 1e-1))
    batch_size: int = 16

    # Build GPU caches once — reused across all trials.
    gpu_train = GPUTensorCache.from_cached(train_cache, device)
    gpu_val = GPUTensorCache.from_cached(val_cache, device)

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", lr_min, lr_max, log=True)
        weight_decay = trial.suggest_float("weight_decay", wd_min, wd_max, log=True)

        _, solver = _build_seg_probe_and_solver(
            model, num_classes, eval_cfg, device, lr=lr, weight_decay=weight_decay
        )
        val_miou = solver.fit_cached(
            train_cache,
            val_cache=val_cache,
            batch_size=batch_size,
            epochs=hpo_epochs,
            verbose=False,
            gpu_train=gpu_train,
            gpu_val=gpu_val,
        )
        return float(val_miou) if val_miou is not None else 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=n_trials)

    best = study.best_params
    logger.info(
        f"[HPO] Best lr={best['lr']:.3e}  weight_decay={best['weight_decay']:.3e}"
        f"  val_mIoU={study.best_value:.4f}  ({n_trials} trials)"
    )
    return {"lr": best["lr"], "weight_decay": best["weight_decay"]}
