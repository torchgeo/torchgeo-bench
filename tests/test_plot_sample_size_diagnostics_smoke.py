import subprocess
import sys

import pandas as pd
import pytest


def test_plot_sample_size_diagnostics_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")

    csv_path = tmp_path / "sample_size.csv"
    rows: list[dict[str, object]] = []
    cls_models = ["resnet50", "dofa", "panopticon"]
    seg_models = ["resnet50", "dofa", "panopticon"]
    datasets = ["m-eurosat", "so2sat"]
    seg_datasets = ["burn_scars", "pastis"]
    fractions = [0.01, 0.10, 0.25]
    seeds = [0, 1]

    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(cls_models):
            for fraction in fractions:
                for seed in seeds:
                    accuracy = (
                        0.50 + 0.15 * (fraction * 4) + 0.03 * (2 - model_idx) - 0.01 * dataset_idx
                    )
                    signed_ece = 0.24 - 0.08 * fraction + 0.02 * model_idx + 0.01 * dataset_idx
                    ece = signed_ece + 0.04
                    mean_wrong_confidence = 0.82 - 0.10 * fraction + 0.03 * model_idx
                    overconfidence_gap = 0.16 - 0.05 * fraction + 0.02 * model_idx
                    for metric_name, metric_value in [
                        ("accuracy", accuracy + 0.002 * seed),
                        ("ece", ece + 0.001 * seed),
                        ("signed_ece", signed_ece + 0.001 * seed),
                        ("mean_wrong_confidence", mean_wrong_confidence + 0.001 * seed),
                        ("overconfidence_gap", overconfidence_gap + 0.001 * seed),
                    ]:
                        rows.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "train_fraction": fraction,
                                "seed": seed,
                                "task": "classification",
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                            }
                        )

    for dataset_idx, dataset in enumerate(seg_datasets):
        for model_idx, model in enumerate(seg_models):
            for fraction in fractions:
                for seed in seeds:
                    miou = 0.28 + 0.18 * fraction + 0.02 * (2 - model_idx) - 0.01 * dataset_idx
                    pixel_ece = 0.20 - 0.06 * fraction + 0.02 * model_idx + 0.005 * dataset_idx
                    for metric_name, metric_value in [
                        ("miou", miou + 0.001 * seed),
                        ("pixel_ece", pixel_ece + 0.001 * seed),
                    ]:
                        rows.append(
                            {
                                "model": model,
                                "dataset": dataset,
                                "train_fraction": fraction,
                                "seed": seed,
                                "task": "segmentation",
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                            }
                        )

    pd.DataFrame(rows).to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/plot_sample_size_diagnostics.py",
            str(csv_path),
            "--outdir",
            str(outdir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    assert (outdir / "classification" / "rank_heatmap_signed_ece.png").exists()
    assert (outdir / "classification" / "low_data_signed_ece.png").exists()
    assert (outdir / "classification" / "scatter" / "accuracy_vs_signed_ece.png").exists()
    assert (outdir / "classification" / "tau_vs_fraction_signed_ece.csv").exists()
    assert (outdir / "segmentation" / "rank_heatmap_miou.png").exists()
    assert (outdir / "summary.md").exists()
