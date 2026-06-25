"""Sample-size sweep analysis.

Consumes the long-format CSV produced by slurm/sample_size_full.sbatch
(`torchgeo-bench sample-size`) and visualizes how performance and calibration
change as the training-set fraction shrinks, plus how the model ranking
reshuffles across data thresholds.

CSV schema (one row per metric):
  model,dataset,train_fraction,seed,task,n_train_full,n_train_used,
  n_val,n_test,best_c,metric_name,metric_value

Analysis is per-task (classification / segmentation have disjoint datasets and
metrics). For each task it emits, into --out-dir:
  - curves_{cls,seg}.png  : metric x dataset grid of metric-vs-fraction curves,
                            one line per model, mean +/- 1 std band over seeds.
  - rank_table_{cls,seg}.{csv,md} : model x fraction aggregate rank
                            (evaluma aggregate_ranking, agg="mean").
  - tau_vs_fraction_{cls,seg}.csv + tau_vs_fraction_{cls,seg}.png :
                            Kendall tau of each fraction's ranking against the
                            full-data (largest fraction) ranking, point estimate
                            (no bootstrap CI -- only 4 datasets/task), plus the
                            per-dataset tau values.
  - tau_matrix_{cls,seg}.png : full pairwise fraction x fraction Kendall-tau heatmap.
  - summary.md            : stitches the tables and figure references together.

Ranking/tau are restricted to the set of models present at *every* fraction
(intersection); dropped models are printed. Curves plot whatever is present.
Re-run as-is whenever more model rows land in the CSV.

Run (after `source sc_venv_template/activate.sh`):
  python experiments/sample_size_analysis.py
  python experiments/sample_size_analysis.py --x n_train
  python experiments/sample_size_analysis.py --csv results/sample_size_full.csv \
      --out-dir experiments/sample_size_figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

EVALUMA_PATH = Path("/p/project1/hai_uqmethodbox/nils/evaluma")
sys.path.insert(0, str(EVALUMA_PATH))
import evaluma  # noqa: E402

# Reuse the shared model metadata (short names, pretraining groups, markers,
# group colors) from the CKA prototypes so styling stays consistent across the
# project. The sample-size CSV's `model` column uses the same canonical names.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torchgeo_bench.cka.prototypes import (  # noqa: E402
    GROUP_COLORS,
    GROUP_ORDER,
    group_for_model,
    marker_for_model,
    short_name,
)

CSV_DEFAULT = Path("results/sample_size_full.csv")
OUT_DIR_DEFAULT = Path("experiments/sample_size_figures")

# Metric -> ("max"/"min", pretty axis label). Higher-is-better unless "min".
METRIC_INFO = {
    "accuracy": ("max", "Accuracy"),
    "miou": ("max", "mIoU"),
    "ece": ("min", "ECE (lower better)"),
    "nll": ("min", "NLL (lower better)"),
    "pixel_ece": ("min", "Pixel ECE (lower better)"),
}
# Display order of metrics within a task.
TASK_METRICS = {
    "classification": ["accuracy", "ece", "nll"],
    "segmentation": ["miou", "pixel_ece"],
}
TASK_SLUG = {"classification": "cls", "segmentation": "seg"}


def df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame (with its index) as a GitHub markdown table.

    Local replacement for DataFrame.to_markdown so we don't depend on tabulate.
    """
    idx_name = df.index.name or ""
    headers = [idx_name] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [f"{v:g}" if isinstance(v, (int, float)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def style_fraction_axis(ax, fractions, xscale: str, base_label: str = "train fraction") -> None:
    """Pin x ticks to the actual fractions as percentages and set the label.

    Appends "(log scale)" to the label only when the axis is actually log, so
    readers aren't misled by the even spacing of geometric ticks.
    """
    log_x = xscale == "log"
    ax.set_xscale("log" if log_x else "linear")
    ticks = sorted(fractions)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{f * 100:g}%" for f in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel(base_label + (" (log scale)" if log_x else ""), fontsize=8)


def load(csv_path: Path) -> pd.DataFrame:
    """Load the sweep CSV, keeping rows with a finite metric value."""
    df = pd.read_csv(csv_path)
    df = df[np.isfinite(df["metric_value"])].copy()
    return df


def seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seeds: one (model, dataset, fraction, metric) -> mean value.

    Also keeps mean n_train_used so the curves can switch to an absolute x-axis.
    """
    grp = df.groupby(["model", "dataset", "train_fraction", "metric_name"], as_index=False)
    return grp.agg(
        metric_value=("metric_value", "mean"),
        metric_std=("metric_value", "std"),
        n_train_used=("n_train_used", "mean"),
    )


# --------------------------------------------------------------------------
# Trend curves
# --------------------------------------------------------------------------


def plot_curves(df: pd.DataFrame, task: str, out_dir: Path, x_mode: str, xscale: str) -> Path:
    """Grid of metric-vs-fraction curves: rows=metric, cols=dataset.

    One line per model, mean over seeds with a +/-1 std shaded band.
    On the fraction axis, ticks are pinned to the actual fractions and labelled
    as percentages regardless of scale.
    """
    metrics = [m for m in TASK_METRICS[task] if m in set(df["metric_name"])]
    datasets = sorted(df["dataset"].unique())
    # Order models by pretraining group so the legend clusters by strategy.
    models = sorted(
        df["model"].unique(),
        key=lambda m: (GROUP_ORDER.index(group_for_model(m))
                       if group_for_model(m) in GROUP_ORDER else len(GROUP_ORDER), m),
    )
    if not metrics or not datasets:
        return None

    # Color encodes pretraining group; marker encodes the individual model.
    def color(m: str):
        return GROUP_COLORS.get(group_for_model(m), "#999999")

    nrows, ncols = len(metrics), len(datasets)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False, sharex=True
    )

    x_col = "n_train_used" if x_mode == "n_train" else "train_fraction"
    fractions = sorted(df["train_fraction"].unique())

    for r, metric in enumerate(metrics):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            sub = df[(df["metric_name"] == metric) & (df["dataset"] == dataset)]
            for model in models:
                ms = sub[sub["model"] == model].sort_values(x_col)
                if ms.empty:
                    continue
                x = ms[x_col].to_numpy()
                y = ms["metric_value"].to_numpy()
                std = ms["metric_std"].fillna(0.0).to_numpy()
                ax.plot(
                    x, y, marker=marker_for_model(model), ms=4, lw=1.3,
                    color=color(model), label=short_name(model),
                )
                ax.fill_between(x, y - std, y + std, color=color(model), alpha=0.12)
            show_xlabel = r == nrows - 1
            if x_mode == "n_train":
                # Counts span orders of magnitude -> always log, default ticks.
                ax.set_xscale("log")
                if show_xlabel:
                    ax.set_xlabel("n train used (log scale)", fontsize=8)
            else:
                style_fraction_axis(ax, fractions, xscale)
                if not show_xlabel:
                    ax.set_xlabel("")
            if r == 0:
                ax.set_title(dataset, fontsize=9)
            if c == 0:
                ax.set_ylabel(METRIC_INFO[metric][1], fontsize=8)
            ax.grid(True, which="both", ls=":", alpha=0.4)
            ax.tick_params(labelsize=7)

    # Per-model legend (marker = model, colored by its group) at the bottom...
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 6),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    # ...plus a compact group -> color key so the pretraining strategy is legible.
    present_groups = [g for g in GROUP_ORDER if any(group_for_model(m) == g for m in models)]
    group_handles = [
        Line2D([0], [0], color=GROUP_COLORS[g], lw=3, label=g) for g in present_groups
    ]
    fig.legend(
        handles=group_handles,
        loc="upper right",
        title="pretraining",
        fontsize=7,
        title_fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.995, 0.995),
    )
    fig.suptitle(f"Sample-size sweep — {task}", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = out_dir / f"curves_{TASK_SLUG[task]}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Ranking + rank sensitivity (evaluma)
# --------------------------------------------------------------------------


def common_models(df: pd.DataFrame, task: str, fractions: list[float]) -> list[str]:
    """Models with a complete (dataset x fraction) grid for *every* task metric.

    evaluma rankings require a full model x dataset score matrix at each
    fraction, so a model missing any (dataset, fraction, metric) cell is dropped
    from the ranking/tau analysis (it still appears in the curves).
    """
    metrics = [m for m in TASK_METRICS[task] if m in set(df["metric_name"])]
    datasets = sorted(df["dataset"].unique())
    required = {(d, f, m) for d in datasets for f in fractions for m in metrics}
    keep = []
    for model in sorted(df["model"].unique()):
        have = set(
            zip(
                *[
                    df[df["model"] == model][c]
                    for c in ("dataset", "train_fraction", "metric_name")
                ],
                strict=False,
            )
        )
        if required <= have:
            keep.append(model)
    return keep


def benchmark_for_fraction(df: pd.DataFrame, fraction: float, metric: str, models: list[str]):
    """Build an evaluma Benchmark (one metric, datasets as columns) at a fraction."""
    sub = df[
        (df["train_fraction"] == fraction)
        & (df["metric_name"] == metric)
        & (df["model"].isin(models))
    ]
    long = sub[["model", "dataset", "metric_name", "metric_value"]].rename(
        columns={"metric_name": "metric", "metric_value": "score"}
    )
    # evaluma keys metric_direction by *dataset column*; every column here shares
    # this one metric, so map all datasets to its direction.
    direction = METRIC_INFO[metric][0]
    metric_direction = {d: direction for d in long["dataset"].unique()}
    return evaluma.load_df(
        long,
        model="model",
        dataset="dataset",
        metric="metric",
        score="score",
        norm_ref_low=0.0,
        norm_ref_high=1.0,
        metric_direction=metric_direction,
    )


def ranks_from_benchmark(bench, models: list[str]) -> pd.Series:
    """Rank 1 = best. Returns a Series indexed by model."""
    table = bench.aggregate_ranking(agg="mean").table  # sorted desc by score
    order = {m: i + 1 for i, m in enumerate(table["model"])}
    return pd.Series({m: order[m] for m in models})


def rank_table(df: pd.DataFrame, task: str, metric: str, fractions: list[float], models: list[str]):
    """model x fraction rank table for one metric. Returns (DataFrame, {frac: Series})."""
    per_frac = {}
    for f in fractions:
        bench = benchmark_for_fraction(df, f, metric, models)
        per_frac[f] = ranks_from_benchmark(bench, models)
    out = pd.DataFrame({f"frac={f:g}": per_frac[f] for f in fractions})
    out = out.sort_values(out.columns[-1])
    # Prepend the pretraining group, and index by short name for readability.
    out.insert(0, "group", [group_for_model(m) for m in out.index])
    out.index = [short_name(m) for m in out.index]
    out.index.name = "model"
    return out, per_frac


def tau_vs_reference(
    df: pd.DataFrame, metric: str, fractions: list[float], models: list[str]
) -> pd.DataFrame:
    """Kendall tau of each fraction's ranking vs. the largest-fraction ranking.

    Aggregate tau (point estimate, no CI -- only ~4 datasets/task) plus the
    per-dataset tau values to show the spread honestly.
    """
    ref_frac = max(fractions)
    datasets = sorted(df[df["metric_name"] == metric]["dataset"].unique())

    # Aggregate ranking per fraction (across datasets), and per-dataset scores.
    agg_ranks = {f: ranks_from_benchmark(benchmark_for_fraction(df, f, metric, models), models)
                 for f in fractions}

    def per_dataset_score(fraction: float, dataset: str) -> pd.Series:
        sub = df[
            (df["train_fraction"] == fraction)
            & (df["metric_name"] == metric)
            & (df["dataset"] == dataset)
            & (df["model"].isin(models))
        ].set_index("model")["metric_value"]
        return sub.reindex(models)

    ref_agg = agg_ranks[ref_frac]
    rows = []
    for f in fractions:
        agg_tau = float(kendalltau(agg_ranks[f].values, ref_agg.values, method="auto").statistic)
        row = {"fraction": f, "tau_aggregate": agg_tau}
        for d in datasets:
            s_f = per_dataset_score(f, d)
            s_ref = per_dataset_score(ref_frac, d)
            sign = 1.0 if METRIC_INFO[metric][0] == "max" else -1.0
            t = kendalltau(sign * s_f.values, sign * s_ref.values, method="auto").statistic
            row[f"tau[{d}]"] = float(t)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_tau_vs_fraction(
    tau_df: pd.DataFrame, task: str, metric: str, out_dir: Path, xscale: str
) -> Path:
    """Line plot: aggregate tau (bold) + per-dataset tau vs. fraction."""
    fig, ax = plt.subplots(figsize=(5, 3.5))
    x = tau_df["fraction"].to_numpy()
    for col in tau_df.columns:
        if col.startswith("tau[") and col.endswith("]"):
            ax.plot(x, tau_df[col], marker=".", lw=0.9, alpha=0.6, label=col[4:-1])
    ax.plot(x, tau_df["tau_aggregate"], marker="o", lw=2.2, color="k", label="aggregate")
    style_fraction_axis(ax, tau_df["fraction"].tolist(), xscale)
    ax.set_ylabel("Kendall tau vs. full data")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(1.0, ls=":", c="grey", alpha=0.5)
    ax.set_title(f"Rank stability — {task} / {metric}", fontsize=10)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = out_dir / f"tau_vs_fraction_{TASK_SLUG[task]}_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_tau_matrix(
    df: pd.DataFrame, task: str, metric: str, fractions: list[float], models: list[str], out_dir: Path
) -> Path:
    """Full pairwise fraction x fraction Kendall-tau heatmap of the rankings."""
    agg_ranks = {f: ranks_from_benchmark(benchmark_for_fraction(df, f, metric, models), models)
                 for f in fractions}
    n = len(fractions)
    mat = np.ones((n, n))
    for i, fi in enumerate(fractions):
        for j, fj in enumerate(fractions):
            mat[i, j] = kendalltau(
                agg_ranks[fi].values, agg_ranks[fj].values, method="auto"
            ).statistic

    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = [f"{f:g}" for f in fractions]
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("train fraction")
    ax.set_ylabel("train fraction")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title(f"Pairwise rank tau — {task} / {metric}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Kendall tau")
    fig.tight_layout()
    path = out_dir / f"tau_matrix_{TASK_SLUG[task]}_{metric}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def analyze_task(
    df_task: pd.DataFrame, task: str, out_dir: Path, x_mode: str, xscale: str
) -> list[str]:
    """Run all analyses for one task; return markdown blocks for summary.md."""
    slug = TASK_SLUG[task]
    md: list[str] = [f"## {task}\n"]

    curves = plot_curves(df_task, task, out_dir, x_mode, xscale)
    if curves is not None:
        md.append(f"![curves]({curves.name})\n")

    fractions = sorted(df_task["train_fraction"].unique())
    models = common_models(df_task, task, fractions)
    all_models = sorted(df_task["model"].unique())
    dropped = sorted(set(all_models) - set(models))
    if dropped:
        note = f"_Ranking restricted to {len(models)} models present at all fractions. Dropped (incomplete): {', '.join(dropped)}._"
        print(f"[{task}] {note}")
        md.append(note + "\n")
    if len(models) < 2 or len(fractions) < 2:
        md.append("_Not enough models/fractions for ranking analysis yet._\n")
        return md

    for metric in TASK_METRICS[task]:
        if metric not in set(df_task["metric_name"]):
            continue
        rt, _ = rank_table(df_task, task, metric, fractions, models)
        rt.to_csv(out_dir / f"rank_table_{slug}_{metric}.csv")
        (out_dir / f"rank_table_{slug}_{metric}.md").write_text(df_to_md(rt))

        tau_df = tau_vs_reference(df_task, metric, fractions, models)
        tau_df.to_csv(out_dir / f"tau_vs_fraction_{slug}_{metric}.csv", index=False)
        tau_png = plot_tau_vs_fraction(tau_df, task, metric, out_dir, xscale)
        mat_png = plot_tau_matrix(df_task, task, metric, fractions, models, out_dir)

        md.append(f"### {metric}\n")
        md.append("Aggregate rank (rank 1 = best) by fraction:\n")
        md.append(df_to_md(rt) + "\n")
        md.append(f"![tau]({tau_png.name}) ![tau matrix]({mat_png.name})\n")

    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--x", choices=["fraction", "n_train"], default="fraction",
                    help="x-axis quantity for trend curves (default: fraction).")
    ap.add_argument("--xscale", choices=["log", "linear"], default="log",
                    help="Scale for the fraction axis; ticks are pinned to the "
                         "actual fractions as %% either way (default: log). "
                         "Ignored for --x n_train (always log).")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load(args.csv)
    df = seed_mean(df)

    summary: list[str] = ["# Sample-size sweep analysis\n", f"Source: `{args.csv}`\n"]
    for task in ("classification", "segmentation"):
        df_task = df[df["metric_name"].isin(TASK_METRICS[task])]
        if df_task.empty:
            continue
        summary += analyze_task(df_task, task, args.out_dir, args.x, args.xscale)

    (args.out_dir / "summary.md").write_text("\n".join(summary))
    print(f"Wrote figures and summary to {args.out_dir}")


if __name__ == "__main__":
    main()
