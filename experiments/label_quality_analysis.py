"""Inspect and visualize the segmentation label-quality audit CSV.

The pipeline (mode=label_quality) emits one tidy row per (model, dataset,
method, sample) into ``results/label_quality_results.csv``. This script turns
that into the three views the audit is actually read through, grouped by model:

  1. a per-(model, dataset, method) summary table — coverage, spread, capacity;
  2. method-agreement stats — Spearman/Kendall over the full ranking plus
     top-N overlap, which is where cleanlab and AER tend to part ways;
  3. figures — score distributions, rank-vs-rank scatter, top-N overlap curve.

Model-aware: terramind variants (``tt_terramind_v1_base``,
``tt_terramind_v1_base_rgb``, …) are merged to a canonical ``terramind``. The
older reference CSV predates the ``model`` column; it is back-filled with
``resnet50`` so this script still runs on it (that CSV was a resnet50-only
sweep), which is exactly how the runnable half of the workflow is validated
before trusted multi-model results exist.

Usage:
    python experiments/label_quality_analysis.py \
        --csv results/label_quality_results.csv \
        --out experiments/label_quality_figures

    # rank table for one (model, dataset), most-suspect first
    python experiments/label_quality_analysis.py --dataset cloudsen12 --top 25
"""

import argparse
import logging
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Rank 1 is most suspect for both methods, so ranks are directly comparable even
# though the underlying scores run in opposite directions (cleanlab: lower is
# suspect; AER: higher is suspect).
TOP_N_GRID = [10, 25, 50, 100, 250, 500]

# The two terramind configs (`..._base`, `..._base_rgb`) are the same model; the
# analysis layer is where raw per-config slugs collapse to a canonical name.
_TERRAMIND_RE = re.compile(r".*terramind.*", re.IGNORECASE)


def canonical_model(name: str) -> str:
    """Canonical model name for grouping: terramind variants → ``terramind``."""
    return "terramind" if _TERRAMIND_RE.match(str(name)) else str(name)


def load(csv_path: str) -> pd.DataFrame:
    """Read the tidy audit CSV, back-filling and canonicalizing ``model``.

    Fails loudly on a missing/empty file. If the CSV predates the ``model``
    column (the resnet50-only reference sweep), synthesizes ``model="resnet50"``
    so the model-aware workflow still runs on it.
    """
    if not os.path.exists(csv_path):
        raise SystemExit(f"No results CSV at {csv_path} — the sweep has not written rows yet.")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise SystemExit(f"{csv_path} is empty.")
    if "model" not in df.columns:
        df["model"] = "resnet50"  # back-compat: the reference CSV was resnet50-only
    df["model"] = df["model"].fillna("resnet50").map(canonical_model)
    return df


def is_degenerate(sub: pd.DataFrame) -> bool:
    """Whether any row of ``sub`` is flagged degenerate.

    Legacy CSVs predate the column entirely; a missing column and a NaN both mean
    *unknown*, which is reported as not-degenerate but is never evidence of
    health. Only an explicit ``True`` suppresses a cell.

    A NaN inside a CSV that *does* carry the column is a different animal: it
    means the file mixes rows from before and after the gate landed, so this
    cell's health was never measured. Silently reading that as healthy is how a
    collapsed cell reaches a headline number, so it is warned about loudly here
    and fixed at the source by ``scripts/clean_label_quality_csv.py``.
    """
    if "degenerate" not in sub.columns:
        return False
    column = sub["degenerate"]
    unknown = int(column.isna().sum())
    if unknown:
        logging.warning(
            "Cell %s: %d/%d rows have an unknown `degenerate` (pre-gate rows in an "
            "append-only CSV). Treating them as not-degenerate, which is NOT evidence "
            "of health -- run scripts/clean_label_quality_csv.py.",
            _cell_label(sub), unknown, len(column),
        )
    # `astype(bool)` on an object column would make the string "False" truthy, so
    # map the CSV's string round-trip explicitly before falling back to fillna.
    flags = column.map(_as_flag)
    return bool(flags.fillna(False).astype(bool).any())


def _as_flag(value) -> bool | float:
    """Coerce one CSV ``degenerate`` cell to a bool, leaving unknowns as NaN."""
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1"}:
            return True
        if text in {"false", "0", ""}:
            return False
        return float("nan")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return float("nan")
    return bool(value)


def _cell_label(sub: pd.DataFrame) -> str:
    """Best-effort "(model, dataset)" label for a warning message."""
    parts = [str(sub[c].iloc[0]) for c in ("model", "dataset") if c in sub.columns and len(sub)]
    return "/".join(parts) if parts else "<unknown>"


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset, method, member_set): coverage, spread, capacity."""
    g = df.groupby(["model", "dataset", "method", "member_set"], dropna=False)
    aggs = {
        "n_samples": ("image_id", "nunique"),
        "score_mean": ("image_score", "mean"),
        "score_std": ("image_score", "std"),
        "score_min": ("image_score", "min"),
        "score_max": ("image_score", "max"),
        "mean_flagged_px": ("n_flagged_pixels", "mean"),
        "low_capacity": ("low_capacity", "first"),
        "tier": ("grouping_tier", "first"),
        "k": ("k", "first"),
        "n_members": ("n_members", "first"),
    }
    # Absent on CSVs written before the degeneracy gate landed.
    for column in ("degenerate", "min_class_coverage", "oof_per_class_iou_min", "score_iqr"):
        if column in df.columns:
            aggs[column] = (column, "first")
    return g.agg(**aggs).reset_index().round(4)


def agreement(df: pd.DataFrame, include_degenerate: bool = False) -> pd.DataFrame:
    """Per-(model, dataset) cleanlab-vs-AER rank agreement, globally and at the top.

    A degenerate cell's ranking is noise, so its correlations are emitted as NaN
    alongside the flag rather than reported as a real method disagreement --
    otherwise a collapsed predictor shows up as a headline "the two methods
    disagree" result. ``include_degenerate`` computes them anyway.
    """
    rows = []
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        wide = sub.pivot_table(index="image_id", columns="method", values="rank")
        if not {"cleanlab", "aer"}.issubset(wide.columns):
            continue
        wide = wide.dropna()
        cl, ae = wide["cleanlab"], wide["aer"]
        degenerate = is_degenerate(sub)
        suppress = degenerate and not include_degenerate
        row = {
            "model": model,
            "dataset": dataset,
            "n": len(wide),
            "degenerate": degenerate,
            "spearman": np.nan if suppress else spearmanr(cl, ae).statistic,
            "kendall": np.nan if suppress else kendalltau(cl, ae).statistic,
        }
        for n in TOP_N_GRID:
            if n > len(wide):
                continue
            a, b = set(cl.nsmallest(n).index), set(ae.nsmallest(n).index)
            row[f"top{n}"] = np.nan if suppress else len(a & b) / n
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def suspect_table(df: pd.DataFrame, model: str, dataset: str, method: str, top: int) -> pd.DataFrame:
    """The ``top`` most-suspect samples for one (model, dataset, method)."""
    sub = df[(df.model == model) & (df.dataset == dataset) & (df.method == method)]
    if sub.empty:
        raise SystemExit(f"No rows for model={model} dataset={dataset} method={method}.")
    cols = ["rank", "image_id", "image_score", "n_flagged_pixels", "fold", "native_id"]
    return sub.nsmallest(top, "rank")[cols].reset_index(drop=True)


def plot_dataset(df: pd.DataFrame, model: str, dataset: str, out_dir: str) -> str | None:
    """Score distributions + rank scatter + top-N overlap for one (model, dataset)."""
    sub = df[(df.model == model) & (df.dataset == dataset)]
    wide = sub.pivot_table(index="image_id", columns="method", values="rank")
    scores = sub.pivot_table(index="image_id", columns="method", values="image_score")
    if not {"cleanlab", "aer"}.issubset(wide.columns):
        return None
    wide, scores = wide.dropna(), scores.dropna()
    degenerate = is_degenerate(sub)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for method, color in (("cleanlab", "#4C6EF5"), ("aer", "#F08C00")):
        axes[0].hist(scores[method], bins=60, alpha=0.6, label=method, color=color)
    axes[0].set_xlabel("image_score")
    axes[0].set_ylabel("samples")
    axes[0].set_title(f"{model} / {dataset}: score distributions")
    axes[0].legend(frameon=False)

    # Suspect corner is bottom-left: rank 1 = most suspect for both methods.
    axes[1].scatter(wide["cleanlab"], wide["aer"], s=6, alpha=0.25, color="#495057")
    axes[1].set_xlabel("cleanlab rank (1 = most suspect)")
    axes[1].set_ylabel("aer rank (1 = most suspect)")
    rho = spearmanr(wide["cleanlab"], wide["aer"]).statistic
    # On a degenerate cell rho measures noise, so it is not presented as agreement.
    axes[1].set_title(
        f"rank agreement — INVALID (Spearman {rho:.2f} on collapsed predictor)"
        if degenerate
        else f"rank agreement (Spearman {rho:.2f})"
    )

    ns = [n for n in TOP_N_GRID if n <= len(wide)]
    ov = [
        len(set(wide["cleanlab"].nsmallest(n).index) & set(wide["aer"].nsmallest(n).index)) / n
        for n in ns
    ]
    axes[2].plot(ns, ov, marker="o", color="#0CA678")
    # Overlap expected if the two rankings were independent.
    axes[2].plot(ns, [n / len(wide) for n in ns], ls="--", color="#ADB5BD", label="chance")
    axes[2].set_xscale("log")
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("top-N")
    axes[2].set_ylabel("overlap fraction")
    axes[2].set_title("cleanlab / AER top-N overlap")
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if degenerate:
        fig.suptitle(
            "DEGENERATE — ranking is noise (OOF members collapsed to the majority class)",
            fontsize=13, color="#C92A2A", fontweight="bold", y=1.04,
        )

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model}_{dataset}_label_quality.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="results/label_quality_v3/label_quality_results.csv")
    ap.add_argument("--out", default="experiments/label_quality_figures")
    ap.add_argument("--model", default=None, help="restrict to one (canonical) model")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--method", default="cleanlab", help="method for the suspect table")
    ap.add_argument("--top", type=int, default=0, help="print the N most-suspect samples")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument(
        "--include-degenerate",
        action="store_true",
        help="report rank agreement for degenerate cells instead of suppressing it",
    )
    args = ap.parse_args()

    df = load(args.csv)
    if args.model:
        df = df[df.model == canonical_model(args.model)]
        if df.empty:
            raise SystemExit(f"No rows for model={args.model}.")
    if args.dataset:
        df = df[df.dataset == args.dataset]
        if df.empty:
            raise SystemExit(f"No rows for dataset={args.dataset}.")

    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("=" * 78)
    print("SUMMARY  (one row per model x dataset x method x member_set)")
    print("=" * 78)
    print(summary_table(df).to_string(index=False))

    agree = agreement(df, include_degenerate=args.include_degenerate)
    if not agree.empty:
        print()
        print("=" * 78)
        print("METHOD AGREEMENT  (cleanlab vs AER; topN = overlap fraction)")
        print("=" * 78)
        print(agree.to_string(index=False))
        if not args.include_degenerate and agree["degenerate"].any():
            print("\nNaN rows above are degenerate cells: their ranking is noise, so the")
            print("correlations are withheld rather than reported as a method disagreement.")
            print("Use --include-degenerate to compute them anyway.")

    # Both quality gates, each naming the metric that triggered it.
    flagged = []
    if "low_capacity" in df.columns:
        lowcap = df[df.low_capacity.fillna(False).astype(bool)]
        flagged += [(m, d, s, "low_capacity (OOF-mIoU below threshold)")
                    for m, d, s in set(zip(lowcap.model, lowcap.dataset, lowcap.member_set))]
    if "degenerate" in df.columns:
        # Via `_as_flag` so a CSV round-trip's literal "False" string does not
        # read as truthy on an object-dtype column.
        degen = df[df.degenerate.map(_as_flag).fillna(False).astype(bool)]
        flagged += [(m, d, s, "degenerate (collapsed predictor / no rank spread)")
                    for m, d, s in set(zip(degen.model, degen.dataset, degen.member_set))]
    if flagged:
        print("\nWARNING: flagged member sets")
        for model, dataset, member_set, why in sorted(flagged):
            print(f"  {model} / {dataset} / {member_set}: {why}")
        print("Rows are kept, never dropped; degenerate cells are excluded from agreement.")

    pairs = sorted(set(zip(df.model, df.dataset)))

    if args.top:
        for model, dataset in pairs:
            print()
            print("=" * 78)
            print(f"TOP {args.top} MOST SUSPECT — {model} / {dataset} / {args.method}")
            print("=" * 78)
            print(suspect_table(df, model, dataset, args.method, args.top).to_string(index=False))

    if not args.no_figures:
        for model, dataset in pairs:
            path = plot_dataset(df, model, dataset, args.out)
            if path:
                print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
