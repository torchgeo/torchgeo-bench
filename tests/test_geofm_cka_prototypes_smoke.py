import subprocess
import sys

import pandas as pd
import pytest

MODELS = [
    "tt_clay_v1_5_base",
    "tgeo_panopticon",
    "resnet50",
    "vit_large_patch16_224",
]
DATASETS = ["advance", "m-eurosat"]
CORRUPTIONS = ["cloud", "motion_blur"]


def _family_params(name: str) -> tuple[float, float, float]:
    if name in {"tt_clay_v1_5_base", "tgeo_panopticon"}:
        return 0.95, 0.03, -0.08
    return 0.82, 0.01, -0.30


def _cka_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for name in MODELS:
            rows.append(
                {
                    "name": name,
                    "dataset": dataset,
                    "corruption_type": "clean",
                    "severity": 0,
                    "layer_name": "head",
                    "layer_index": 4,
                    "cka": 1.0,
                    "spearman_drift_confidence": float("nan"),
                    "frac_overconfident_high_drift": 0.0,
                }
            )
            base_cka, base_overconf, base_coupling = _family_params(name)
            for corruption in CORRUPTIONS:
                for severity in range(1, 6):
                    rows.append(
                        {
                            "name": name,
                            "dataset": dataset,
                            "corruption_type": corruption,
                            "severity": severity,
                            "layer_name": "head",
                            "layer_index": 4,
                            "cka": base_cka - 0.018 * severity,
                            "spearman_drift_confidence": base_coupling - 0.01 * severity,
                            "frac_overconfident_high_drift": base_overconf + 0.01 * severity,
                        }
                    )
    return rows


def _uq_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for name in MODELS:
            eo_family = name in {"tt_clay_v1_5_base", "tgeo_panopticon"}
            for corruption in ["clean", *CORRUPTIONS]:
                severities = [0] if corruption == "clean" else list(range(1, 6))
                for severity in severities:
                    signed_ece = 0.01 if corruption == "clean" else (0.03 if eo_family else -0.02) + 0.02 * severity
                    accuracy = 0.94 if corruption == "clean" else (0.88 if eo_family else 0.84) - 0.06 * severity
                    for metric_name, metric_value in [
                        ("signed_ece", signed_ece),
                        ("accuracy", accuracy),
                    ]:
                        rows.append(
                            {
                                "model": "synthetic",
                                "name": name,
                                "backbone": name,
                                "dataset": dataset,
                                "normalization": "none",
                                "image_size": 224,
                                "interpolation": "bilinear",
                                "partition": "default",
                                "bands": "rgb",
                                "seed": 42,
                                "svgp_n_inducing": None,
                                "svgp_pca_dim": None,
                                "svgp_zca": None,
                                "svgp_kernel": None,
                                "svgp_epochs": None,
                                "svgp_lr": None,
                                "svgp_inducing_init": None,
                                "svgp_mixing_weights": None,
                                "uq_method": "uncalibrated",
                                "corruption_type": corruption,
                                "severity": severity,
                                "metric_name": metric_name,
                                "metric_value": metric_value,
                                "n_cal": 10,
                                "n_train": 10,
                                "n_test": 10,
                                "best_c": 1.0,
                                "feature_dim": 4,
                                "trace_dataset_root": "",
                                "trace_run_id": "",
                                "trace_block_key": "",
                            }
                        )
    return rows


def _trace_rows(name: str, dataset: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eo_family = name in {"tt_clay_v1_5_base", "tgeo_panopticon"}
    for corruption in CORRUPTIONS:
        for severity in range(1, 6):
            for sample_idx in range(12):
                drift = 0.45 * severity + 0.04 * sample_idx + (0.12 if eo_family else 0.0)
                wrong = sample_idx >= 8
                confidence = 0.92 - 0.01 * severity if eo_family else 0.82 - 0.04 * severity
                if not wrong:
                    confidence += 0.06
                rows.append(
                    {
                        "corruption_type": corruption,
                        "severity": severity,
                        "sample_idx": sample_idx,
                        "drift": drift,
                        "confidence": max(0.05, min(0.999, confidence)),
                        "correct": not wrong,
                        "y_true": sample_idx % 3,
                        "y_pred": (sample_idx + int(wrong)) % 3,
                        "logits": [1.0, 0.5, -0.5],
                    }
                )
    return rows


def test_geofm_cka_prototypes_smoke(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("pyarrow")

    cka_csv = tmp_path / "cka_results.csv"
    uq_csv = tmp_path / "uq_focused_results.csv"
    traces_root = tmp_path / "cka_traces"
    pd.DataFrame(_cka_rows()).to_csv(cka_csv, index=False)
    pd.DataFrame(_uq_rows()).to_csv(uq_csv, index=False)

    for name in MODELS:
        model_dir = traces_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        for dataset in DATASETS:
            pd.DataFrame(_trace_rows(name, dataset)).to_parquet(model_dir / f"{dataset}.parquet", index=False)

    outdir = tmp_path / "figs"
    cmd = [
        sys.executable,
        "viz/geofm_cka_prototypes.py",
        "--cka-csv",
        str(cka_csv),
        "--uq-csv",
        str(uq_csv),
        "--traces-root",
        str(traces_root),
        "--outdir",
        str(outdir),
        "--format",
        "png",
        "--datasets",
        ",".join(DATASETS),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    expected = [
        "proto1_family_scatter_signed_ece.png",
        "proto1_family_scatter_facets_signed_ece.png",
        "proto1_group_scatter_signed_ece.png",
        "proto1_family_scatter_accuracy.png",
        "proto1_family_scatter_facets_accuracy.png",
        "proto1_group_scatter_accuracy.png",
        "proto1_family_scatter_by_corruption_signed_ece.png",
        "proto1_group_scatter_by_corruption_signed_ece.png",
        "proto1_family_scatter_facets_signed_ece_cloud.png",
        "proto1_family_scatter_facets_signed_ece_motion_blur.png",
        "proto1_family_scatter_wrong_confidence.png",
        "proto1_family_scatter_facets_wrong_confidence.png",
        "proto1_group_scatter_wrong_confidence.png",
        "proto2_quadrant_scatter.png",
        "proto2_quadrant_density.png",
        "proto2_quadrant_severity.png",
        "proto3_dataset_dumbbell.png",
        "proto3_dataset_effects.png",
        "proto5_severity_trajectory.png",
        "proto5_severity_by_dataset.png",
        "proto5_severity_by_corruption.png",
        "proto6_quadrant_bars.png",
        "proto6_quadrant_stacked.png",
        "proto6_quadrant_by_dataset.png",
    ]
    for filename in expected:
        path = outdir / filename
        assert path.exists(), f"missing {filename}: {result.stderr}"
        assert path.stat().st_size > 0
