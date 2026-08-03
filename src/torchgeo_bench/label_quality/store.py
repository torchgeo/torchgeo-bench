"""Persistence for the label-quality pipeline.

Four artifact kinds share the ``results/label_quality/<dataset>/`` root:

- the tidy ``label_quality_results.csv`` (one row per sample per method), written
  through the shared atomic appender so parallel dataset runs are lock-safe;
- per-sample ``.npz`` artifacts (pixel score map + predicted mask + join
  metadata) under ``<dataset>/<model_slug>/<method>/<image_id>.npz`` for the
  audit gallery;
- fold-model checkpoints keyed ``(dataset, model_slug, member_idx, fold_idx,
  seed)`` — the granularity at which the orchestrator resumes. The
  ``model_slug`` segment isolates each backbone so a resume never cross-loads
  another model's weights (FM-1);
- an optional ``<dataset>/<model_slug>/training_curves.json`` recording each
  member's train loss / held-out mIoU / per-class IoU over training, written
  only when ``label_quality.eval_every`` is set.

Two independent quality gates annotate rows rather than dropping them, so
downstream ranking can down-weight instead of silently discarding:

- ``low_capacity`` — OOF macro-mIoU below ``LOW_CAPACITY_THRESHOLD``. A scalar
  capacity check, and structurally blind to a majority-class collapse: AER's
  macro-IoU is per image over the classes present in that image's label, so a
  background-only predictor averages to ≈0.5 and clears a 0.3 threshold while
  predicting nothing but background.
- ``degenerate`` — the gate that does catch it (see :mod:`.degeneracy`), backed
  by ``min_class_coverage`` / ``oof_per_class_iou_min`` (cell-level, identical
  across a cell's two methods) and ``score_iqr`` (per method). Rows so flagged
  carry a ranking that is noise and must be excluded from agreement statistics.
"""

import json
import logging
import os

import numpy as np
import torch

from ..segmentation_viz import save_segmentation_viz

logger = logging.getLogger(__name__)

# Ordered CSV schema — one row per (sample, method).
CSV_COLUMNS = [
    "dataset",
    "model",
    "method",
    "member_set",
    "image_id",
    "image_score",
    "rank",
    "grouping_tier",
    "fold",
    "n_flagged_pixels",
    "k",
    "n_members",
    "seed",
    "bands",
    "partition",
    "low_capacity",
    "native_id",
    # Degeneracy gate (appended so the order stays additive for older readers).
    "degenerate",
    "min_class_coverage",
    "oof_per_class_iou_min",
    "score_iqr",
]
# Zero-pad width for ``image_id`` (e.g. sample 42 -> "00042").
_IMAGE_ID_WIDTH = 5
# OOF macro-mIoU below this marks a member set as low-capacity.
LOW_CAPACITY_THRESHOLD = 0.3


def image_id(index: int) -> str:
    """Zero-padded string id for a sample index."""
    return str(int(index)).zfill(_IMAGE_ID_WIDTH)


def sanitize_slug(name: str) -> str:
    """Path-safe slug for a model name: ``/`` (and whitespace) collapse to ``-``.

    Used to key checkpoints and npz artifacts by model identity. The raw
    per-config name is kept (no alias collapsing); terramind canonicalization is
    a viz-layer concern, so runs on disk stay isolated per config.
    """
    return "-".join(str(name).split()).replace("/", "-")


def write_results_csv(
    output_path: str,
    *,
    dataset: str,
    model: str,
    method: str,
    member_set: str,
    image_scores,
    n_flagged_pixels,
    grouping_tier: str,
    folds,
    native_ids,
    k: int,
    n_members: int,
    seed: int,
    bands: str,
    partition: str,
    low_capacity: bool,
    degenerate: bool = False,
    min_class_coverage: float = float("nan"),
    oof_per_class_iou_min: float = float("nan"),
    score_iqr: float = float("nan"),
    lower_is_suspect: bool = True,
) -> None:
    """Append one tidy row per sample to the label-quality results CSV.

    Args:
        output_path: Destination CSV (created if missing).
        dataset: Dataset name.
        model: Model slug (raw ``cfg.model.name``) the members were built from.
        method: Scoring method (``"cleanlab"`` / ``"aer"``).
        member_set: Label describing the ensemble configuration.
        image_scores: Per-sample image scores ``(N,)``.
        n_flagged_pixels: Per-sample flagged-pixel counts ``(N,)``.
        grouping_tier: Fold-assignment tier used for this dataset.
        folds: Per-sample held-out fold id ``(N,)``.
        native_ids: Per-sample native ids, or ``None`` when unavailable.
        k: Number of folds.
        n_members: Number of ensemble members.
        seed: Base seed for the run.
        bands: Band configuration string.
        partition: Dataset partition name.
        low_capacity: Whether this member set is low-capacity (annotated on every row).
        degenerate: Whether this cell's ranking is noise (collapsed predictor or
            spreadless scores). Defaults to ``False`` rather than NaN so pandas
            keeps a bool dtype instead of coercing the column to object.
        min_class_coverage: Cell-level GT-relative predicted mass of the rarest
            present class, minimised over members (identical across methods).
        oof_per_class_iou_min: Cell-level global per-class IoU minimum.
        score_iqr: IQR of *this method's* image-score distribution (per method).
        lower_is_suspect: If ``True`` (Cleanlab), lower scores rank first; if
            ``False`` (AER), higher scores rank first. Rank 1 is most suspect.
    """
    image_scores = np.asarray(image_scores, dtype=np.float64)
    n_flagged_pixels = np.asarray(n_flagged_pixels)
    folds = np.asarray(folds)
    ranks = _ranks(image_scores, lower_is_suspect)
    # Cell constants: coerce once, not per sample.
    degenerate = bool(degenerate)
    min_class_coverage = float(min_class_coverage)
    oof_per_class_iou_min = float(oof_per_class_iou_min)
    score_iqr = float(score_iqr)

    rows = []
    for i in range(len(image_scores)):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "member_set": member_set,
                "image_id": image_id(i),
                "image_score": float(image_scores[i]),
                "rank": int(ranks[i]),
                "grouping_tier": grouping_tier,
                "fold": int(folds[i]),
                "n_flagged_pixels": int(n_flagged_pixels[i]),
                "k": int(k),
                "n_members": int(n_members),
                "seed": int(seed),
                "bands": bands,
                "partition": partition,
                "low_capacity": bool(low_capacity),
                "native_id": "" if native_ids is None else str(native_ids[i]),
                "degenerate": degenerate,
                "min_class_coverage": min_class_coverage,
                "oof_per_class_iou_min": oof_per_class_iou_min,
                "score_iqr": score_iqr,
            }
        )

    from ..main import append_rows_atomic  # lazy: avoids main<->run<->store cycle

    append_rows_atomic(output_path, rows)


def _ranks(scores: np.ndarray, lower_is_suspect: bool) -> np.ndarray:
    """Dense 1..N ranks with rank 1 = most suspect."""
    order = np.argsort(scores if lower_is_suspect else -scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def _artifact_dir(root: str, dataset: str, model_slug: str, method: str) -> str:
    """``<root>/label_quality/<dataset>/<model_slug>/<method>`` (created).

    Keyed by ``model_slug`` so a multi-model sweep never overwrites another
    backbone's per-sample artifacts (mirrors the checkpoint layout).
    """
    dest = os.path.join(root, "label_quality", dataset, model_slug, method)
    os.makedirs(dest, exist_ok=True)
    return dest


def save_pixel_artifact(
    root: str,
    dataset: str,
    model_slug: str,
    method: str,
    image_id_str: str,
    pixel_scores: np.ndarray,
    pred_mask: np.ndarray,
    *,
    native_id: str = "",
    image_score: float | None = None,
    rank: int | None = None,
) -> str:
    """Write a per-sample ``.npz`` for the audit gallery; return its path.

    Contents: ``pixel_scores`` (H,W score map), ``pred_mask`` (H,W argmax), plus
    the join metadata the gallery reloads the source image by (``native_id``,
    ``image_id``) and annotates each panel with (``image_score``, ``rank``).
    """
    path = os.path.join(_artifact_dir(root, dataset, model_slug, method), f"{image_id_str}.npz")
    np.savez(
        path,
        pixel_scores=pixel_scores,
        pred_mask=pred_mask,
        native_id=np.asarray(native_id),
        image_id=np.asarray(image_id_str),
        image_score=np.asarray(np.nan if image_score is None else image_score),
        rank=np.asarray(-1 if rank is None else rank),
    )
    return path


def save_training_curves(
    root: str, dataset: str, model_slug: str, curves: list[dict], **metadata
) -> str:
    """Write the per-member training curves for one (dataset, model) as JSON.

    Per-member evidence, across all K×M trainings, that the members behaved as
    the hyperparameter search predicted — train loss, held-out mIoU and per-class
    IoU sampled every ``label_quality.eval_every`` steps
    (``docs/plans/segmentation_hparam_search.md`` D11). Written only when curves
    were recorded; with ``eval_every`` unset the launch produces none.

    Folds restored from a checkpoint on a resume contribute no curve, so a
    resumed run's file can cover fewer than K×M trainings.
    """
    dest = os.path.join(root, "label_quality", dataset, model_slug)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "training_curves.json")
    with open(path, "w") as f:
        json.dump({**metadata, "dataset": dataset, "model": model_slug, "curves": curves}, f, indent=2)
    return path


def save_per_class_diagnostics(
    root: str, dataset: str, model_slug: str, cell: dict, **metadata
) -> str:
    """Write one cell's per-class degeneracy detail as JSON.

    The CSV carries only the two *minima* (``min_class_coverage``,
    ``oof_per_class_iou_min``), which say that a cell collapsed but not which
    classes died — and that distinction is the whole diagnosis: collapse
    concentrated in the rarest class while common classes hold up points at the
    training objective, whereas uniformly mediocre IoU points at capacity or the
    probing head instead.

    JSON rather than new CSV columns because the vectors are variable length
    (2–13 present classes across these datasets) and ``coverage`` is 2-D
    ``(members × classes)``; neither fits a fixed tidy schema.

    Args:
        root: Results root (the directory holding the results CSV).
        dataset: Dataset name.
        model_slug: Sanitized model slug.
        cell: The dict returned by :func:`degeneracy.cell_metrics`.
        **metadata: Extra scalars to record alongside (e.g. ``num_classes``).

    Returns:
        The path written.
    """
    dest = os.path.join(root, "label_quality", dataset, model_slug)
    os.makedirs(dest, exist_ok=True)
    path = os.path.join(dest, "per_class_iou.json")
    payload = {
        **metadata,
        "dataset": dataset,
        "model": model_slug,
        **{k: _jsonable(v) for k, v in cell.items()},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def _jsonable(value):
    """Coerce numpy scalars/arrays to JSON-serializable Python types.

    ``json`` cannot encode ``np.float64`` or ``ndarray``, and NaN round-trips
    through ``json`` as the non-standard ``NaN`` literal, which Python's own
    decoder reads back correctly — the diagnostic is read by this repo only.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def load_pixel_artifact(
    root: str, dataset: str, model_slug: str, method: str, image_id_str: str
) -> dict:
    """Reload a per-sample artifact as a dict of its stored arrays."""
    path = os.path.join(
        root, "label_quality", dataset, model_slug, method, f"{image_id_str}.npz"
    )
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def checkpoint_path(
    root: str, dataset: str, *, model_slug: str, member_idx: int, fold_idx: int, seed: int
) -> str:
    """Resume-anchor path encoding ``(dataset, model_slug, member_idx, fold_idx, seed)``.

    The ``model_slug`` segment keeps each backbone's checkpoints in their own
    subtree, so a resume never cross-loads another model's weights (FM-1).
    """
    fname = f"member{member_idx}_fold{fold_idx}_seed{seed}.pt"
    return os.path.join(root, "label_quality", dataset, "checkpoints", model_slug, fname)


def checkpoint_exists(
    root: str, dataset: str, *, model_slug: str, member_idx: int, fold_idx: int, seed: int
) -> bool:
    """Whether the fold-model checkpoint already exists (drives resume skip)."""
    return os.path.exists(
        checkpoint_path(
            root,
            dataset,
            model_slug=model_slug,
            member_idx=member_idx,
            fold_idx=fold_idx,
            seed=seed,
        )
    )


def save_checkpoint(
    state, root: str, dataset: str, *, model_slug: str, member_idx: int, fold_idx: int, seed: int
) -> str:
    """Persist a fold-model ``state`` under its resume key; return the path."""
    path = checkpoint_path(
        root,
        dataset,
        model_slug=model_slug,
        member_idx=member_idx,
        fold_idx=fold_idx,
        seed=seed,
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    return path


def is_low_capacity(oof_miou: float, threshold: float = LOW_CAPACITY_THRESHOLD) -> bool:
    """Whether an out-of-fold macro-mIoU is below the low-capacity threshold."""
    return bool(oof_miou < threshold)


def save_overlays(
    root: str,
    dataset: str,
    method: str,
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    num_classes: int,
    rgb_indices: list[int],
    ignore_index: int = 255,
    n_samples: int = 8,
) -> None:
    """Render inspection overlays for top-ranked samples via ``segmentation_viz``."""
    save_segmentation_viz(
        out_dir=os.path.join(root, "label_quality", dataset, "overlays"),
        model_name=method,
        dataset_name=dataset,
        images=images,
        gt_masks=gt_masks,
        pred_masks=pred_masks,
        num_classes=num_classes,
        rgb_indices=rgb_indices,
        ignore_index=ignore_index,
        n_samples=n_samples,
    )
