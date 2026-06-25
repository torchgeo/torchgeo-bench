"""Confidence distribution violin plots for the sample-size sweep.

For each classification dataset, produces one figure with 5 rows (one per
train fraction, 0.01 → 0.75 top-to-bottom).  Within each row there is one
pair of violins per model (correct predictions in blue, wrong in vermillion),
models grouped by pretraining strategy: EO-MAE | EO-DINO | Nat-DINO | Nat-sup.

Only a representative ImageNet-pretrained subset is shown:
  ViT-L/16, ResNet-50, ConvNeXt-L, Swin-Tiny
All EO-pretrained models are shown.  RCF is excluded.

Colors follow the Wong colorblind-safe palette:
  correct → #0072B2 (blue)
  wrong   → #D55E00 (vermillion)

Data source: results/sample_size_image_stats/task=classification/
  (Hive-partitioned Parquet, one file per model/dataset/fraction/seed)

Run (after `source sc_venv_template/activate.sh`):
  python experiments/sample_size_violin.py
  python experiments/sample_size_violin.py --out-dir experiments/sample_size_figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torchgeo_bench.cka.prototypes import (  # noqa: E402
    GROUP_COLORS,
    GROUP_ORDER,
    group_for_model,
    short_name,
)

DATA_ROOT = Path("results/sample_size_image_stats/task=classification")
OUT_DIR_DEFAULT = Path("experiments/sample_size_figures")

# EO models: all available
EO_MODELS = [
    "tt_clay_v1_5_base",
    "tt_prithvi_eo_v2_300_tl",
    "tgeo_dofa_base",
    "olmoearth_v1_1_base",
    "olmoearth_v1_1_tiny",
    "tt_terramind_v1_base_rgb",
    "tgeo_panopticon",
    "vit_large_patch16_dinov3sat",
]

# ImageNet representative subset (one per arch family, no RCF)
IMAGENET_MODELS = [
    "convnext_large_dinov3",
    "vit_large_patch16_224",
    "resnet50",
    "swin_tiny_patch4_window7_224",
]

ALL_MODELS = EO_MODELS + IMAGENET_MODELS

CORRECT_COLOR = "#0072B2"
WRONG_COLOR = "#D55E00"

TRAIN_FRACTIONS = [0.01, 0.1, 0.25, 0.5, 0.75]


def load_data(data_root: Path) -> pd.DataFrame:
    """Load confidence + correctness columns for the selected models only.

    Reads per-model subdirectories directly to avoid pyarrow schema-merge
    issues that arise from large_string vs string divergence across files.
    """
    COLS = ["dataset", "train_fraction", "confidence", "correct"]
    chunks: list[pd.DataFrame] = []
    model_set = set(ALL_MODELS)
    for model_dir in data_root.iterdir():
        # directory name: model=<name>
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name.removeprefix("model=")
        if model_name not in model_set:
            continue
        for pq_file in model_dir.rglob("*.parquet"):
            chunk = pd.read_parquet(pq_file, columns=COLS)
            chunk["model"] = model_name
            chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    df["correct"] = df["correct"].astype(bool)
    return df


def model_order() -> list[str]:
    """Return ALL_MODELS sorted by GROUP_ORDER, preserving within-group order."""
    def sort_key(m: str) -> tuple[int, int]:
        g = group_for_model(m)
        g_idx = GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
        return (g_idx, ALL_MODELS.index(m))
    return sorted(ALL_MODELS, key=sort_key)


def group_boundaries(ordered_models: list[str]) -> list[tuple[int, int, str]]:
    """Return list of (start_idx, end_idx_exclusive, group_label) spans."""
    spans: list[tuple[int, int, str]] = []
    prev_group = None
    start = 0
    for i, m in enumerate(ordered_models):
        g = group_for_model(m)
        if g != prev_group:
            if prev_group is not None:
                spans.append((start, i, prev_group))
            start = i
            prev_group = g
    if prev_group is not None:
        spans.append((start, len(ordered_models), prev_group))
    return spans


def draw_violin(ax: plt.Axes, x: float, values: np.ndarray, color: str, width: float = 0.35) -> None:
    """Draw a single violin at position x."""
    if len(values) < 2:
        ax.scatter([x], [values.mean() if len(values) else 0.5], color=color, s=10, zorder=3)
        return
    parts = ax.violinplot(
        [values],
        positions=[x],
        widths=width,
        showmedians=True,
        showextrema=False,
    )
    for pc_key in ("bodies",):
        for body in parts[pc_key]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.75)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(1.2)


def plot_dataset(df_ds: pd.DataFrame, dataset_name: str, out_dir: Path) -> None:
    ordered = model_order()
    # Filter to models actually present in this dataset
    present = set(df_ds["model"].unique())
    ordered = [m for m in ordered if m in present]
    if not ordered:
        return

    n_models = len(ordered)
    n_fracs = len(TRAIN_FRACTIONS)

    # x positions: pair of violins per model, gap between groups
    spans = group_boundaries(ordered)
    gap = 1.2  # extra x-space between groups
    x_correct = {}
    x_wrong = {}
    x_tick = {}
    x_cursor = 0.0
    pair_width = 0.8
    prev_end = 0
    for span_start, span_end, _ in spans:
        if span_start > 0:
            x_cursor += gap
        for local_i, m in enumerate(ordered[span_start:span_end]):
            global_i = span_start + local_i
            cx = x_cursor + global_i * pair_width * 2 - span_start * pair_width * 2
            x_correct[m] = cx - pair_width * 0.3
            x_wrong[m] = cx + pair_width * 0.3
            x_tick[m] = cx
        prev_end = span_end

    # Recompute properly: iterate with accumulated offset
    x_correct = {}
    x_wrong = {}
    x_tick = {}
    offset = 0.0
    for span_idx, (span_start, span_end, _) in enumerate(spans):
        if span_idx > 0:
            offset += gap
        for local_i in range(span_end - span_start):
            m = ordered[span_start + local_i]
            cx = offset + local_i * pair_width * 2
            x_correct[m] = cx - pair_width * 0.3
            x_wrong[m] = cx + pair_width * 0.3
            x_tick[m] = cx
        offset += (span_end - span_start) * pair_width * 2

    fig_width = max(12, n_models * 1.2 + 2)
    fig, axes = plt.subplots(
        n_fracs, 1,
        figsize=(fig_width, 2.5 * n_fracs),
        sharey=True,
        sharex=True,
    )
    if n_fracs == 1:
        axes = [axes]

    for row_idx, frac in enumerate(TRAIN_FRACTIONS):
        ax = axes[row_idx]
        df_frac = df_ds[np.isclose(df_ds["train_fraction"], frac)]

        for m in ordered:
            df_m = df_frac[df_frac["model"] == m]
            correct_vals = df_m.loc[df_m["correct"], "confidence"].values
            wrong_vals = df_m.loc[~df_m["correct"], "confidence"].values

            draw_violin(ax, x_correct[m], correct_vals, CORRECT_COLOR, width=pair_width * 0.55)
            draw_violin(ax, x_wrong[m], wrong_vals, WRONG_COLOR, width=pair_width * 0.55)

        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{int(frac * 100)}%\ntrain", fontsize=13, rotation=0, labelpad=50, va="center")
        ax.yaxis.set_label_position("left")
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", lw=0.4, alpha=0.4)
        ax.set_facecolor("#f8f8f8")

        # Group dividers
        for span_idx, (span_start, span_end, group_label) in enumerate(spans):
            if span_idx > 0:
                boundary_x = (
                    x_wrong[ordered[span_start - 1]] + x_correct[ordered[span_start]]
                ) / 2
                ax.axvline(boundary_x, color="#aaaaaa", lw=0.8, ls="--")
            # Group label on top row only
            if row_idx == 0:
                group_x = np.mean([x_tick[ordered[i]] for i in range(span_start, span_end)])
                ax.text(
                    group_x, 1.08, group_label,
                    ha="center", va="bottom", fontsize=13,
                    fontweight="bold",
                    color=GROUP_COLORS.get(group_label, "#444444"),
                    transform=ax.get_xaxis_transform(),
                )

    # Shared x-axis ticks (model short names)
    axes[-1].set_xticks([x_tick[m] for m in ordered])
    axes[-1].set_xticklabels(
        [short_name(m) for m in ordered],
        rotation=35, ha="right", fontsize=12,
    )
    axes[-1].tick_params(axis="x", length=0)

    # Legend
    legend_handles = [
        mpatches.Patch(color=CORRECT_COLOR, label="Correct"),
        mpatches.Patch(color=WRONG_COLOR, label="Wrong"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        fontsize=13,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(f"{dataset_name} — confidence by correctness", fontsize=14, y=1.01)
    fig.text(
        -0.01, 0.5, "Confidence (max softmax prob)",
        va="center", rotation="vertical", fontsize=13,
    )

    out_path = out_dir / f"violin_confidence_{dataset_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading parquet data...")
    df = load_data(args.data_root)
    print(f"  {len(df):,} rows, {df['dataset'].nunique()} datasets, {df['model'].nunique()} models")

    for dataset_name in sorted(df["dataset"].unique()):
        print(f"Plotting {dataset_name}...")
        plot_dataset(df[df["dataset"] == dataset_name].copy(), dataset_name, args.out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
