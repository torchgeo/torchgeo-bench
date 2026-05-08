#!/usr/bin/env python
"""Plot two C-sweep gap heatmaps from results/c_sweep_experiment.csv.

Chart 1: test_acc[best_val_C] - test_acc[C=1]
   How much does picking C on the validation set beat just leaving C at the
   default of 1.0? Positive cells = val-tuning helps. Negative cells = the
   default happens to do as well (often within noise on small splits).

Chart 2: test_acc[best_val_C] - test_acc[best_train_C]
   How wrong would you be if you picked C from the training set instead of
   the validation set? Positive cells = val-picking beats train-picking.
   train_acc is monotone in C, so train-picked C typically saturates at the
   largest C in the grid.

Both charts share a symmetric RdBu_r colour scale so they're directly
comparable. Cells are annotated with the gap in percentage points.

Usage:
    python scripts/plot_c_sweep_gaps.py
    python scripts/plot_c_sweep_gaps.py --input results/c_sweep_experiment.csv \\
                                        --output-dir figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = "results/c_sweep_experiment.csv"
DEFAULT_OUTDIR = "figures"
C_DEFAULT = 1.0


def compute_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, model), return one row with all the C-pick comparisons."""
    rows: list[dict] = []
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        g = g.sort_values("C").reset_index(drop=True)
        bv = g.loc[g["val_acc"].idxmax()]
        bt = g.loc[g["train_acc"].idxmax()]
        c1 = g.loc[(g["C"] - C_DEFAULT).abs().idxmin()]
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "test_at_best_val": float(bv["test_acc"]),
                "test_at_best_train": float(bt["test_acc"]),
                "test_at_C1": float(c1["test_acc"]),
                "best_val_C": float(bv["C"]),
                "best_train_C": float(bt["C"]),
                "C1_actual": float(c1["C"]),
                "gap_default_vs_best_val": float(bv["test_acc"] - c1["test_acc"]),
                "gap_train_vs_val_pick": float(bv["test_acc"] - bt["test_acc"]),
            }
        )
    return pd.DataFrame(rows)


def _model_order(gaps: pd.DataFrame) -> list[str]:
    """Sort models by mean test_acc@best_val across datasets (low → high)."""
    return gaps.groupby("model")["test_at_best_val"].mean().sort_values().index.tolist()


def _plot_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str = "RdBu_r",
) -> None:
    """Render one signed-gap heatmap with numeric annotations."""
    data = matrix.to_numpy() * 100  # -> percentage points
    im = ax.imshow(data, cmap=cmap, vmin=vmin * 100, vmax=vmax * 100, aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title, fontsize=11, pad=8)

    # Cell annotations: dark text on light cells, white on saturated cells.
    halfspan = max(abs(vmin), abs(vmax)) * 100
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = data[r, c]
            colour = "white" if abs(v) > 0.6 * halfspan else "black"
            ax.text(
                c,
                r,
                f"{v:+.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=colour,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Δ test accuracy (pp)", fontsize=9)


def make_charts(gaps: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    """Render both heatmaps and return their output paths."""
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = sorted(gaps["dataset"].unique())
    models = _model_order(gaps)

    g1 = gaps.pivot(index="dataset", columns="model", values="gap_default_vs_best_val").loc[
        datasets, models
    ]
    g2 = gaps.pivot(index="dataset", columns="model", values="gap_train_vs_val_pick").loc[
        datasets, models
    ]

    span = max(g1.abs().max().max(), g2.abs().max().max())
    vmin, vmax = -span, span

    out1 = outdir / "c_sweep_gap_default_vs_best_val.png"
    out2 = outdir / "c_sweep_gap_train_vs_val_pick.png"

    for matrix, out, title in [
        (
            g1,
            out1,
            "C-sweep: test acc gain from picking C on val vs using C=1\n"
            "(positive ⇒ validation-tuning beats the default)",
        ),
        (
            g2,
            out2,
            "C-sweep: test acc gain from picking C on val vs picking on train\n"
            "(positive ⇒ validation-picking beats train-picking)",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        _plot_heatmap(ax, matrix, title=title, vmin=vmin, vmax=vmax)
        ax.set_xlabel("model (← lower mean test acc · higher mean test acc →)")
        ax.set_ylabel("dataset")
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return out1, out2


def _summary(name: str, gap_col: str, gaps: pd.DataFrame) -> str:
    """One-line stdout summary per chart."""
    g = gaps[gap_col] * 100  # pp
    best_idx = g.idxmax()
    worst_idx = g.idxmin()
    return (
        f"{name}: mean={g.mean():+.2f}pp, median={g.median():+.2f}pp, "
        f"max={g.max():+.2f}pp ({gaps.loc[best_idx, 'dataset']}/{gaps.loc[best_idx, 'model']}), "
        f"min={g.min():+.2f}pp ({gaps.loc[worst_idx, 'dataset']}/{gaps.loc[worst_idx, 'model']}), "
        f"|gap|>1pp in {(g.abs() > 1).sum()}/{len(g)} cells"
    )


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output-dir", default=DEFAULT_OUTDIR, type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"error: input CSV not found: {args.input}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.input)
    required = {"dataset", "model", "C", "train_acc", "val_acc", "test_acc"}
    missing = required - set(df.columns)
    if missing:
        print(f"error: input CSV missing columns: {sorted(missing)}", file=sys.stderr)
        return 2

    gaps = compute_gaps(df)
    print(f"Computed gaps for {len(gaps)} (dataset, model) combinations.")
    print(_summary("Gap1 (best_val vs C=1)         ", "gap_default_vs_best_val", gaps))
    print(_summary("Gap2 (best_val vs best_train)  ", "gap_train_vs_val_pick", gaps))

    out1, out2 = make_charts(gaps, args.output_dir)
    print(f"Wrote {out1}")
    print(f"Wrote {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
