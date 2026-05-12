#!/usr/bin/env python
"""Per-class breakdown of cleanlab issues for multi-label datasets.

For each multi-label dataset (benv2, m-bigearthnet, treesatai), runs
``cleanlab.multilabel_classification.filter.find_multilabel_issues_per_class``
on the saved test probs and reports per-class statistics:

- ``n_pos``: number of positives in the given labels
- ``n_pos_pred``: number of positives predicted at the 0.5 threshold
- ``n_flagged``: how many samples cleanlab flagged for that class
- ``flag_rate``: ``n_flagged / N`` (overall) and ``n_flagged_pos / n_pos``
- ``ap``: per-class average precision (sklearn) from probs
- ``co_occur_top``: top class index whose given/pred patterns most overlap
  with this class — used to spot near-identical / nested classes that the
  model can't distinguish

Output: ``results/cleanlab/perclass_<dataset>_<split>.csv`` and a printed
top-K worst classes per dataset.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("perclass")


def _ap_per_class(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-class average precision; NaN for classes with zero positives."""
    from sklearn.metrics import average_precision_score

    out = np.full(y.shape[1], np.nan, dtype=np.float64)
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        out[c] = float(average_precision_score(y[:, c], p[:, c]))
    return out


def _per_class_flags(y: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Return (N, K) boolean mask — cleanlab flag for each (sample, class)."""
    from cleanlab.multilabel_classification.filter import find_multilabel_issues_per_class

    y_lists = [np.flatnonzero(row).tolist() for row in y]
    out = find_multilabel_issues_per_class(labels=y_lists, pred_probs=probs)
    if isinstance(out, tuple):
        out = out[0]
    return np.asarray(out, dtype=bool)


def _co_occur_top(y_col: np.ndarray, y_other: np.ndarray) -> tuple[int, float]:
    """For column ``c``, find the column that maximizes Jaccard with it."""
    n_other = y_other.shape[1]
    best_j, best_score = -1, 0.0
    a = y_col.astype(bool)
    for j in range(n_other):
        b = y_other[:, j].astype(bool)
        inter = int((a & b).sum())
        union = int((a | b).sum())
        if union == 0:
            continue
        s = inter / union
        if s > best_score:
            best_score, best_j = s, j
    return best_j, best_score


def report_dataset(npz_path: Path, out_dir: Path, top_k: int = 10) -> pd.DataFrame:
    z = np.load(npz_path, allow_pickle=True)
    y = z["labels"].astype(np.int64)
    probs = z["probs"].astype(np.float32)
    if y.ndim != 2:
        raise SystemExit(f"{npz_path}: not multi-label (labels ndim={y.ndim})")
    K = y.shape[1]
    pred_pos = (probs > 0.5).astype(np.int64)
    aps = _ap_per_class(y, probs)
    flags = _per_class_flags(y, probs)

    N = y.shape[0]
    rows = []
    for c in range(K):
        f = flags[:, c]
        n_pos = int(y[:, c].sum())
        n_pred = int(pred_pos[:, c].sum())
        n_flag = int(f.sum())
        # Flag rate restricted to true positives — "of the samples labeled c,
        # how many does cleanlab think are wrong?"
        flag_pos = int((f & y[:, c].astype(bool)).sum())
        flag_neg = int((f & ~y[:, c].astype(bool)).sum())
        # Highest Jaccard with another class (excluding self) — proxy for
        # near-duplicate / nested labels.
        mask = np.ones(K, dtype=bool)
        mask[c] = False
        j_idx, j_score = _co_occur_top(y[:, c], y[:, mask])
        # Map j_idx back through the mask to original column index.
        real_idx = int(np.flatnonzero(mask)[j_idx]) if j_idx >= 0 else -1
        rows.append(
            {
                "class": c,
                "n_pos": n_pos,
                "n_pred_pos": n_pred,
                "ap": aps[c],
                "n_flagged": n_flag,
                "flag_rate": n_flag / max(len(f), 1),
                "flag_among_pos": flag_pos / max(n_pos, 1),
                "flag_among_neg": flag_neg / max(N - n_pos, 1),
                "jaccard_top_class": real_idx,
                "jaccard_top_score": j_score,
            }
        )
    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir
        / f"perclass_{npz_path.stem.replace('__', '_').rsplit('_', 1)[0]}_{npz_path.stem.rsplit('_', 1)[1]}.csv"
    )
    # Above messes up; do it cleaner:
    stem = npz_path.stem  # e.g. benv2__tt_terramind_v1_large_test
    dataset, rest = stem.split("__", 1)
    split = rest.rsplit("_", 1)[1]  # train|test
    out_path = out_dir / f"perclass_{dataset}_{split}.csv"
    df.to_csv(out_path, index=False)
    logger.warning("[%s/%s] K=%d wrote %s", dataset, split, K, out_path)

    # Print worst classes by `flag_among_pos`.
    worst = df.sort_values("flag_among_pos", ascending=False).head(top_k)
    print(f"\n=== {dataset}/{split} — top {top_k} classes by flag_among_pos ===")
    print(
        worst.to_string(
            index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        )
    )

    # Also print classes with high Jaccard (near-duplicate labels).
    sticky = df.sort_values("jaccard_top_score", ascending=False).head(top_k)
    print(f"\n=== {dataset}/{split} — top {top_k} classes by Jaccard with another class ===")
    print(
        sticky[
            ["class", "jaccard_top_class", "jaccard_top_score", "n_pos", "ap", "flag_among_pos"]
        ].to_string(
            index=False, float_format=lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
        )
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probs-dir",
        type=Path,
        default=Path("results/cleanlab/probs"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("results/cleanlab"))
    parser.add_argument("--datasets", nargs="*", default=["benv2", "m-bigearthnet", "treesatai"])
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    for ds in args.datasets:
        for split in args.splits:
            cands = sorted(args.probs_dir.glob(f"{ds}__*_{split}.npz"))
            if not cands:
                logger.warning("No probs for %s/%s", ds, split)
                continue
            for p in cands:
                try:
                    report_dataset(p, args.out_dir, top_k=args.top_k)
                except Exception:
                    logger.exception("[%s] failed", p)


if __name__ == "__main__":
    main()
