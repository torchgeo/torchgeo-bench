import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pandas as pd
import pytest


def _load_plot_uq_results_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_uq_results.py"
    spec = spec_from_file_location("plot_uq_results_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_dataset_trend_grid_includes_clean_and_uses_name():
    module = _load_plot_uq_results_module()
    df, _ = module._normalize_frame(  # noqa: SLF001
        pd.DataFrame(
            [
                {
                    "dataset": "m-eurosat",
                    "uq_method": "laplace",
                    "corruption_type": "cloud",
                    "severity": 2,
                    "metric_name": "ece",
                    "metric_value": 0.2,
                    "name": "resnet50",
                    "model": "m.t",
                },
                {
                    "dataset": "m-eurosat",
                    "uq_method": "uncalibrated",
                    "corruption_type": "clean",
                    "severity": 0,
                    "metric_name": "ece",
                    "metric_value": 0.1,
                    "name": "resnet18",
                    "model": "m.t",
                },
                {
                    "dataset": "m-eurosat",
                    "uq_method": "conformal",
                    "corruption_type": "cloud",
                    "severity": 2,
                    "metric_name": "empirical_coverage",
                    "metric_value": 0.9,
                    "name": "convnext_large_dinov3",
                    "model": "m.t",
                },
            ]
        )
    )

    subset, methods, models, severities = module._prepare_dataset_trend_grid(  # noqa: SLF001
        df,
        dataset="m-eurosat",
        corruption_type="cloud",
        metrics=["ece", "empirical_coverage"],
    )

    assert sorted(subset["severity_int"].astype(int).unique().tolist()) == [0, 2]
    assert methods == ["uncalibrated", "laplace", "conformal"]
    assert models == ["convnext_large_dinov3", "resnet18", "resnet50"]
    assert severities == [0, 2]


def test_plot_uq_reliability_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    csv_path = tmp_path / "uq_results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "m-eurosat",
                "uq_method": "uncalibrated",
                "corruption_type": "clean",
                "severity": 0,
                "metric_name": "ece",
                "metric_value": 0.1,
                "name": "resnet50",
                "model": "m.t",
            }
        ]
    ).to_csv(csv_path, index=False)

    trace_run_dir = tmp_path / "uq_traces" / "run_id=run-1"
    trace_run_dir.mkdir(parents=True)

    trace_path = trace_run_dir / "part-000.csv"
    pd.DataFrame(
        {
            "dataset": ["m-eurosat"] * 4,
            "backbone": ["resnet50"] * 4,
            "uq_method": ["uncalibrated"] * 4,
            "corruption_type": ["clean"] * 4,
            "severity": [0] * 4,
            "confidence": [0.9, 0.7, 0.6, 0.2],
            "correct": [1, 1, 0, 0],
            "sample_idx": [0, 1, 2, 3],
        }
    ).to_csv(trace_path, index=False)

    manifest_path = trace_run_dir / "manifest.csv"
    pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "model": "m.t",
                "backbone": "resnet50",
                "name": "resnet50",
                "dataset": "m-eurosat",
                "partition": "default",
                "bands": "rgb",
                "normalization": "bandspec_zscore",
                "image_size": 224,
                "interpolation": "bilinear",
                "uq_method": "uncalibrated",
                "corruption_type": "clean",
                "severity": 0,
                "seed": 42,
                "trace_path": str(trace_path),
                "trace_format": "csv",
                "n_test": 4,
                "schema_version": "v1",
                "created_at_utc": "2026-05-13T00:00:00Z",
                "config_hash": "abc",
                "git_sha": "",
            }
        ]
    ).to_csv(manifest_path, index=False)

    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/plot_uq_results.py",
        str(csv_path),
        "--outdir",
        str(outdir),
        "--trace-dir",
        str(trace_run_dir),
        "--format",
        "png",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    reliability_dir = outdir / "reliability"
    files = list(reliability_dir.glob("*.png"))
    assert files


def test_plot_uq_dataset_trends_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    csv_path = tmp_path / "uq_results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "m-eurosat",
                "uq_method": "uncalibrated",
                "corruption_type": "clean",
                "severity": 0,
                "metric_name": "ece",
                "metric_value": 0.1,
                "name": "resnet18",
                "model": "m.t",
            },
            {
                "dataset": "m-eurosat",
                "uq_method": "uncalibrated",
                "corruption_type": "cloud",
                "severity": 1,
                "metric_name": "ece",
                "metric_value": 0.2,
                "name": "resnet18",
                "model": "m.t",
            },
            {
                "dataset": "m-eurosat",
                "uq_method": "temp_scaling",
                "corruption_type": "clean",
                "severity": 0,
                "metric_name": "ece",
                "metric_value": 0.08,
                "name": "resnet50",
                "model": "m.t",
            },
            {
                "dataset": "m-eurosat",
                "uq_method": "temp_scaling",
                "corruption_type": "cloud",
                "severity": 1,
                "metric_name": "ece",
                "metric_value": 0.16,
                "name": "resnet50",
                "model": "m.t",
            },
            {
                "dataset": "m-eurosat",
                "uq_method": "conformal",
                "corruption_type": "clean",
                "severity": 0,
                "metric_name": "empirical_coverage",
                "metric_value": 0.92,
                "name": "convnext_large_dinov3",
                "model": "m.t",
            },
            {
                "dataset": "m-eurosat",
                "uq_method": "conformal",
                "corruption_type": "cloud",
                "severity": 1,
                "metric_name": "empirical_coverage",
                "metric_value": 0.89,
                "name": "convnext_large_dinov3",
                "model": "m.t",
            },
        ]
    ).to_csv(csv_path, index=False)

    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/plot_uq_results.py",
        str(csv_path),
        "--outdir",
        str(outdir),
        "--corruptions",
        "cloud",
        "--metrics",
        "ece,empirical_coverage",
        "--calibration-metric",
        "",
        "--format",
        "png",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    out_path = outdir / "by_dataset_trends" / "m_eurosat__cloud.png"
    assert out_path.exists()
