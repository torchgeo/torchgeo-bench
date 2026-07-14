"""Aggregate a CoordBench results CSV into per-family mean-rank leaderboards.

Proper aggregation for a benchmark suite whose datasets live on wildly different
scales and reward different specialties:

1. **Mean rank, not mean metric.** Within each benchmark, models are ranked
   (1 = best); the reported number is the mean of those ranks. This is robust to
   datasets with very different metric ranges (a single high-variance R² can't
   dominate a raw mean).
2. **By task family.** Benchmarks are grouped into ``socio`` / ``env`` / ``land``
   (see :mod:`torchgeo_bench.coordbench.taxonomy`) because location encoders
   specialize; one macro number hides that.
3. **By holdout.** ``random`` and ``spatial`` cross-validation are reported
   separately (spatial is the harder, geographically-disjoint test). Benchmarks
   with an official fixed split contribute to both views.

Multi-task benchmarks are collapsed to one score (mean over tasks, R² floored)
before ranking, so each benchmark counts once.

Usage: ``python -m torchgeo_bench.coordbench.leaderboard <results.csv> [--method linear|knn]``
"""

import argparse

import pandas as pd

from torchgeo_bench.coordbench.taxonomy import FAMILIES, R2_FLOOR, family_of


def _bench_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (model, benchmark) to one score: mean over tasks, R² floored."""
    d = df.copy()
    is_r2 = d["metric_name"] == "r2"
    d.loc[is_r2, "metric_value"] = d.loc[is_r2, "metric_value"].clip(lower=R2_FLOOR)
    scores = (
        d.groupby(["model_name", "dataset"], as_index=False)["metric_value"]
        .mean()
        .rename(columns={"metric_value": "score"})
    )
    scores["family"] = scores["dataset"].map(family_of)
    return scores


def _leaderboard(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (mean-rank, mean-metric) tables: rows = models, cols = families + macro."""
    # Rank models within each benchmark (1 = best; ties share the average rank).
    scores = scores.copy()
    scores["rank"] = scores.groupby("dataset")["score"].rank(ascending=False, method="average")
    models = sorted(scores["model_name"].unique())

    def _pivot(value: str) -> pd.DataFrame:
        out = {}
        for fam in FAMILIES:
            sub = scores[scores["family"] == fam]
            out[f"{fam} (n={sub['dataset'].nunique()})"] = (
                sub.groupby("model_name")[value].mean().reindex(models)
            )
        out["macro"] = scores.groupby("model_name")[value].mean().reindex(models)
        return pd.DataFrame(out)

    return _pivot("rank"), _pivot("score")


def main() -> None:
    """Print per-family mean-rank + mean-metric tables for random and spatial holdouts."""
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--method", default="linear", help="probe method label (e.g. linear, knn5)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    method_col = df["method"].astype(str)
    df = df[method_col.str.startswith(args.method.rstrip("0123456789"))]
    if df.empty:
        raise SystemExit(f"no rows for method={args.method!r} in {args.csv}")

    for view in ("random", "spatial"):
        # Official-split benchmarks (fixed holdout) contribute to both views.
        sub = df[df["split"].isin([view, "official"])]
        if sub.empty:
            continue
        scores = _bench_scores(sub)
        ranks, metrics = _leaderboard(scores)
        order = ranks["macro"].sort_values().index
        print(
            f"\n{'=' * 70}\n{view.upper()} holdout — {args.method} probe — MEAN RANK (lower=better)\n{'=' * 70}"
        )
        with pd.option_context("display.float_format", lambda v: f"{v:6.2f}"):
            print(ranks.reindex(order))
        print(f"\n{view.upper()} holdout — {args.method} probe — mean metric (R²/acc, per family)")
        with pd.option_context("display.float_format", lambda v: f"{v:6.3f}"):
            print(metrics.reindex(order))
    # Surface anything unmapped so the taxonomy stays honest.
    unknown = sorted({d for d in df["dataset"].unique() if family_of(d) == "other"})
    if unknown:
        print(f"\n[warn] benchmarks with no family (counted only in macro): {unknown}")


if __name__ == "__main__":
    main()
