import subprocess
import sys

import pandas as pd
import pytest


def test_plot_uq_error_curves_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    trace_run_dir = (
        tmp_path
        / "uq_traces"
        / "dataset=m-eurosat"
        / "backbone=resnet50"
        / "uq_method=uncalibrated"
        / "corruption_type=clean"
        / "severity=0"
    )
    trace_run_dir.mkdir(parents=True)

    trace_path = trace_run_dir / "trace_block_key=block-1.parquet"
    pd.DataFrame(
        {
            "trace_block_key": ["block-1"] * 6,
            "run_id": ["run-1"] * 6,
            "config_hash": ["abc"] * 6,
            "git_sha": [""] * 6,
            "created_at_utc": ["2026-05-13T00:00:00Z"] * 6,
            "model": ["m.t"] * 6,
            "dataset": ["m-eurosat"] * 6,
            "backbone": ["resnet50"] * 6,
            "partition": ["default"] * 6,
            "bands": ["rgb"] * 6,
            "normalization": ["bandspec_zscore"] * 6,
            "image_size": [224] * 6,
            "interpolation": ["bilinear"] * 6,
            "uq_method": ["uncalibrated"] * 6,
            "corruption_type": ["clean"] * 6,
            "severity": [0] * 6,
            "seed": [42] * 6,
            "sample_id": [f"s{i}" for i in range(6)],
            "is_error": [0, 0, 1, 1, 0, 1],
            "confidence": [0.9, 0.8, 0.4, 0.2, 0.7, 0.1],
            "normalized_predictive_entropy": [0.1, 0.2, 0.7, 0.9, 0.3, 1.0],
            "sample_idx": [0, 1, 2, 3, 4, 5],
            "y_true": [0, 0, 1, 1, 0, 1],
            "y_pred": [0, 0, 0, 0, 0, 0],
            "max_probability": [0.9, 0.8, 0.4, 0.2, 0.7, 0.1],
            "predictive_entropy": [0.1, 0.2, 0.7, 0.9, 0.3, 1.0],
            "correct": [1, 1, 0, 0, 1, 0],
        }
    ).to_parquet(trace_path, index=False)

    outdir = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/plot_uq_error_curves.py",
        "--trace-dir",
        str(tmp_path / "uq_traces"),
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
