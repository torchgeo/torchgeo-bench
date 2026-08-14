# ruff: noqa: I001
#!/usr/bin/env python3
"""Plot each DEO modality against the best historical linear probe."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    """Generate the modality-explicit DEO linear-probe comparison artifacts."""
    deo = pd.concat(
        [
            pd.read_csv(RESULTS / "deo_rgb_classification.csv"),
            pd.read_csv(RESULTS / "deo_s2_classification.csv"),
        ],
        ignore_index=True,
    )
    deo = deo.loc[deo["method"].eq("linear")].copy()
    deo["modality"] = np.where(deo["name"].str.endswith("_rgb"), "RGB", "S2")

    historical = pd.read_csv(RESULTS / "all_results.csv")
    historical = historical.loc[
        historical["method"].eq("linear")
        & historical["metric_value"].notna()
        & ~historical["name"].str.startswith("tgeo_deo_", na=False)
    ].copy()
    best_idx = historical.groupby(["dataset", "metric_name"])["metric_value"].idxmax()
    best = historical.loc[
        best_idx,
        ["dataset", "metric_name", "name", "metric_value", "ci_lower", "ci_upper"],
    ].rename(
        columns={
            "name": "best_model",
            "metric_value": "best_value",
            "ci_lower": "best_ci_lower",
            "ci_upper": "best_ci_upper",
        }
    )

    comparison = deo.merge(best, on=["dataset", "metric_name"], validate="many_to_one")
    comparison = comparison.rename(
        columns={
            "name": "deo_model",
            "metric_value": "deo_value",
            "ci_lower": "deo_ci_lower",
            "ci_upper": "deo_ci_upper",
        }
    )
    comparison["delta"] = comparison["deo_value"] - comparison["best_value"]
    comparison = comparison.sort_values(["metric_name", "dataset", "modality"])
    comparison.to_csv(RESULTS / "deo_modalities_vs_best_linear.csv", index=False)

    metric_titles = {"accuracy": "Accuracy", "micro_mAP": "Micro-mAP"}
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)
    for ax, metric in zip(axes, metric_titles, strict=True):
        data = comparison.loc[comparison["metric_name"].eq(metric)].copy()
        datasets = data["dataset"].drop_duplicates().tolist()
        positions = np.arange(len(datasets))
        values = {dataset: i for i, dataset in enumerate(datasets)}

        baseline = data.drop_duplicates("dataset").set_index("dataset").loc[datasets]
        ax.barh(
            positions + 0.25,
            baseline["best_value"],
            height=0.22,
            color="#9aa0a6",
            label="Best historical linear probe",
            xerr=np.vstack(
                [
                    baseline["best_value"] - baseline["best_ci_lower"],
                    baseline["best_ci_upper"] - baseline["best_value"],
                ]
            ),
            error_kw={"capsize": 3, "elinewidth": 1},
        )
        for modality, color, offset in [("RGB", "#4477aa", 0.0), ("S2", "#ee7733", -0.25)]:
            subset = data.loc[data["modality"].eq(modality)]
            y = np.array([values[dataset] for dataset in subset["dataset"]]) + offset
            ax.barh(
                y,
                subset["deo_value"],
                height=0.22,
                color=color,
                label=f"DEO {modality}",
                xerr=np.vstack(
                    [
                        subset["deo_value"] - subset["deo_ci_lower"],
                        subset["deo_ci_upper"] - subset["deo_value"],
                    ]
                ),
                error_kw={"capsize": 3, "elinewidth": 1},
            )
        ax.set(yticks=positions, yticklabels=datasets, xlim=(0, 1), xlabel=metric_titles[metric])
        ax.set_title(metric_titles[metric])
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="lower right")
        ax.invert_yaxis()

    fig.suptitle("DEO RGB and S2 linear probes versus best historical baseline", fontsize=16)
    fig.savefig(ROOT / "viz" / "deo_modalities_vs_best_linear.png", dpi=160)


if __name__ == "__main__":
    main()
