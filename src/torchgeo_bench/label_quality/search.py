"""Shared construction for the segmentation hyperparameter search.

The search (``docs/plans/segmentation_hparam_search.md``) must train with *the
exact code the launch runs* — a standalone driver reimplementing the loop would
diverge silently in optimizer construction, AMP scaler handling, ``model.train()``
placement, or seed order, invalidating transfer of the hyperparameters it picks
(D11). This module therefore contains no training loop at all. It only:

- composes a Hydra config the way ``main.py`` does (:func:`compose_cfg`),
- builds member specs through :func:`~torchgeo_bench.label_quality.run.build_member_specs`
  and members through :func:`~torchgeo_bench.label_quality.predictors.build_member`
  (:func:`build_search_member`),
- reproduces the launch's *data* setup: the same band selection, the same
  spatially-grouped folds from :func:`~torchgeo_bench.label_quality.folds.assign_folds`,
  and the same fold-relabelling permutation (:func:`fold_split`).

Actual fitting goes through ``Predictor.solver.fit(...)``, i.e.
:meth:`SegmentationSolver._fit_fixed_budget`, which is what the launch calls.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset

from ..datasets.loading import get_bench_dataset_class
from .folds import assign_folds
from .predictors import build_member
from .run import _resolve_band_names, build_member_specs

logger = logging.getLogger(__name__)

# Config dir lives inside the package: src/torchgeo_bench/conf
CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def compose_cfg(model: str, overrides: list[str] | None = None):
    """Compose the project's Hydra config for ``model`` (e.g. ``"timm/resnet50"``).

    Args:
        model: Model config path under ``conf/model`` without the ``.yaml``.
        overrides: Extra Hydra override strings, applied after ``model=``.

    Returns:
        The composed :class:`~omegaconf.DictConfig`.
    """
    with initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return compose(
            config_name="config",
            overrides=[f"model={model}", *(overrides or [])],
        )


def load_bench(dataset: str):
    """Instantiate the registered :class:`BenchDataset` for ``dataset``."""
    return get_bench_dataset_class(dataset)()


def load_train_split(cfg, bench):
    """The train split loaded with exactly the bands the launch uses.

    Mirrors ``_run_one_dataset``: band selection goes through
    :func:`_resolve_band_names` so the loader's channel count matches the
    backbone, and ``lat``/``lon`` metadata is requested so the spatial fold
    tier is available.
    """
    band_names = _resolve_band_names(cfg, bench)
    return bench.get_dataset("train", bands=band_names, metadata=["lat", "lon"])


def load_val_split(cfg, bench):
    """The GeoBench ``val`` split, loaded with the same bands as the train split.

    Used only by the reporting phase (``docs/plans/segmentation_val_phase.md``),
    which trains on the *full* train split at an already-adopted config and
    evaluates once on val. Unlike :func:`load_train_split` this requests no
    ``lat``/``lon`` metadata: there is no fold assignment to compute, and val is
    never partitioned.

    Deliberately not reachable from the trial loop or from :mod:`.oof`: the
    search selects on the grouped hold-out fold, and Cleanlab needs an
    out-of-fold softmax for the *train* samples it audits, so a val loader has
    no place in either.
    """
    band_names = _resolve_band_names(cfg, bench)
    return bench.get_dataset("val", bands=band_names)


def fold_split(bench, cfg, *, k: int, seed: int, fold_idx: int):
    """Train/held-out global indices for one spatially-grouped fold.

    Reproduces the launch's fold construction exactly: ids come from
    :func:`~torchgeo_bench.label_quality.folds.assign_folds` (spatial-block
    grouping at ``grid_cell_deg``), then are relabelled by the same per-member
    permutation ``run.py`` applies, so a member's fold ``fold_idx`` here is the
    fold it would be in the launch.

    A plain random split is deliberately *not* offered: it reports materially
    higher mIoU than the grouped hold-out the pipeline uses, which would tune
    toward a number the launch cannot reproduce (D3).

    Returns:
        ``(train_idx, hold_idx, tier)``.
    """
    lq = cfg.label_quality
    fold_ids, tier = assign_folds(
        bench, "train", k, cell_deg=float(lq.grid_cell_deg), seed=seed
    )
    assignment = np.random.default_rng(seed).permutation(k)[fold_ids]
    hold_idx = np.where(assignment == fold_idx)[0]
    train_idx = np.where(assignment != fold_idx)[0]
    return train_idx, hold_idx, tier


def subset_loader(dataset, indices, batch_size: int, *, shuffle: bool, num_workers: int = 0):
    """Batched loader over a global-index subset, matching ``oof._subset_loader``."""
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_search_member(
    cfg,
    bench,
    *,
    num_classes: int,
    device: str,
    seed: int,
    backbone_lr: float | None = None,
    head_lr: float | None = None,
    max_steps: int | None = None,
):
    """Build one label-quality member, optionally overriding the searched hparams.

    Goes through :func:`build_member_specs` (so the per-model ``layers`` /
    ``head_type`` merge, band resolution and normalization are identical to the
    launch) and then :func:`build_member`, which constructs the unfrozen probe
    and the :class:`SegmentationSolver`. The trial's ``backbone_lr`` / ``head_lr``
    / ``max_steps`` are applied to the resulting spec, so nothing about how the
    member is constructed differs from a launch member except those values.
    """
    specs = build_member_specs(
        cfg, bench, num_classes=num_classes, device=device, seeds=[seed]
    )
    spec = specs[0]
    if backbone_lr is not None:
        spec["backbone_lr"] = float(backbone_lr)
    if head_lr is not None:
        spec["head_lr"] = float(head_lr)
    if max_steps is not None:
        spec["max_steps"] = int(max_steps)
    return build_member(spec)


def free_cuda() -> None:
    """Drop cached CUDA blocks between trials so one trial's peak isn't another's OOM."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
