import subprocess
import sys

import pandas as pd
import pytest


def test_plot_uq_error_curves_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    trace_run_dir = tmp_path / "uq_traces" / "run_id=run-1"
    trace_run_dir.mkdir(parents=True)

    trace_path = trace_run_dir / "part-000.csv"
    pd.DataFrame(
        {
            "dataset": ["m-eurosat"] * 6,
            "backbone": ["resnet50"] * 6,
            "uq_method": ["uncalibrated"] * 6,
            "corruption_type": ["clean"] * 6,
            "severity": [0] * 6,
            "is_error": [0, 0, 1, 1, 0, 1],
            "confidence": [0.9, 0.8, 0.4, 0.2, 0.7, 0.1],
            "normalized_predictive_entropy": [0.1, 0.2, 0.7, 0.9, 0.3, 1.0],
            "sample_idx": [0, 1, 2, 3, 4, 5],
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
                "n_test": 6,
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
        "scripts/plot_uq_error_curves.py",
        "--trace-dir",
        str(trace_run_dir),
        "--outdir",
        str(outdir),
        "--format",
        "png",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    summary_path = outdir / "error_curve_summary.csv"
    assert summary_path.exists()
    summary = pd.read_csv(summary_path)
    assert not summary.empty
