"""Persistence for the label-quality pipeline.

Three artifact kinds share the ``results/label_quality/<dataset>/`` root:

- the tidy ``label_quality_results.csv`` (one row per sample per method), written
  through the shared atomic appender so parallel dataset runs are lock-safe;
- per-sample ``.npz`` artifacts (pixel score map + predicted mask) under
  ``<dataset>/<method>/<image_id>.npz`` for inspection;
- fold-model checkpoints keyed ``(dataset, member_idx, fold_idx, seed)`` — the
  granularity at which the orchestrator resumes.

A member set with weak out-of-fold capacity does not invalidate a run: its rows
are annotated ``low_capacity=True`` (never dropped) so downstream ranking can
down-weight rather than silently discard them.
"""

import logging
import os

import numpy as np
import torch

from ..segmentation_viz import save_segmentation_viz

logger = logging.getLogger(__name__)

# Ordered CSV schema — one row per (sample, method).
CSV_COLUMNS = [
    "dataset",
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
]
# Zero-pad width for ``image_id`` (e.g. sample 42 -> "00042").
_IMAGE_ID_WIDTH = 5
# OOF macro-mIoU below this marks a member set as low-capacity.
LOW_CAPACITY_THRESHOLD = 0.3


def image_id(index: int) -> str:
    """Zero-padded string id for a sample index."""
    return str(int(index)).zfill(_IMAGE_ID_WIDTH)


def write_results_csv(
    output_path: str,
    *,
    dataset: str,
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
    lower_is_suspect: bool = True,
) -> None:
    """Append one tidy row per sample to the label-quality results CSV.

    Args:
        output_path: Destination CSV (created if missing).
        dataset: Dataset name.
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
        lower_is_suspect: If ``True`` (Cleanlab), lower scores rank first; if
            ``False`` (AER), higher scores rank first. Rank 1 is most suspect.
    """
    image_scores = np.asarray(image_scores, dtype=np.float64)
    n_flagged_pixels = np.asarray(n_flagged_pixels)
    folds = np.asarray(folds)
    ranks = _ranks(image_scores, lower_is_suspect)

    rows = []
    for i in range(len(image_scores)):
        rows.append(
            {
                "dataset": dataset,
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


def _artifact_dir(root: str, dataset: str, method: str) -> str:
    """``<root>/label_quality/<dataset>/<method>`` (created)."""
    dest = os.path.join(root, "label_quality", dataset, method)
    os.makedirs(dest, exist_ok=True)
    return dest


def save_pixel_artifact(
    root: str,
    dataset: str,
    method: str,
    image_id_str: str,
    pixel_scores: np.ndarray,
    pred_mask: np.ndarray,
) -> str:
    """Write a per-sample ``.npz`` (pixel score map + predicted mask); return its path."""
    path = os.path.join(_artifact_dir(root, dataset, method), f"{image_id_str}.npz")
    np.savez(path, pixel_scores=pixel_scores, pred_mask=pred_mask)
    return path


def load_pixel_artifact(
    root: str, dataset: str, method: str, image_id_str: str
) -> tuple[np.ndarray, np.ndarray]:
    """Reload a per-sample artifact as ``(pixel_scores, pred_mask)``."""
    path = os.path.join(root, "label_quality", dataset, method, f"{image_id_str}.npz")
    data = np.load(path)
    return data["pixel_scores"], data["pred_mask"]


def checkpoint_path(root: str, dataset: str, *, member_idx: int, fold_idx: int, seed: int) -> str:
    """Resume-anchor path encoding ``(dataset, member_idx, fold_idx, seed)``."""
    fname = f"member{member_idx}_fold{fold_idx}_seed{seed}.pt"
    return os.path.join(root, "label_quality", dataset, "checkpoints", fname)


def checkpoint_exists(root: str, dataset: str, *, member_idx: int, fold_idx: int, seed: int) -> bool:
    """Whether the fold-model checkpoint already exists (drives resume skip)."""
    return os.path.exists(
        checkpoint_path(root, dataset, member_idx=member_idx, fold_idx=fold_idx, seed=seed)
    )


def save_checkpoint(
    state, root: str, dataset: str, *, member_idx: int, fold_idx: int, seed: int
) -> str:
    """Persist a fold-model ``state`` under its resume key; return the path."""
    path = checkpoint_path(root, dataset, member_idx=member_idx, fold_idx=fold_idx, seed=seed)
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
