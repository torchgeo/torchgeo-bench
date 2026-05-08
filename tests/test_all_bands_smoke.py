"""End-to-end smoke test for the all-bands code path.

Runs ``torchgeo-bench run model=timm/resnet18 dataset.bands=all`` on a small
``m-eurosat`` partition and asserts the resulting CSV records the new
``bands`` column with the right value, plus both KNN-5 and linear-probe rows.

Marked ``slow`` because it shells out and runs feature extraction on real data.
"""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.slow
def test_all_bands_e2e(geobench_root, tmp_path: Path):
    """Run torchgeo-bench end-to-end with ``dataset.bands=all`` and check the CSV."""
    del geobench_root  # only used to gate skipping
    cli = shutil.which("torchgeo-bench")
    if cli is None:
        pytest.skip("torchgeo-bench CLI not on PATH")

    output = tmp_path / "results.csv"
    cmd = [
        cli,
        "run",
        "model=timm/resnet18",
        "dataset.names=[m-eurosat]",
        "dataset.bands=all",
        "dataset.partition=0.01x_train",
        "dataset.batch_size=16",
        "eval.bootstrap=10",
        "eval.c_range=[-2,2,3]",
        "device=cpu",
        f"output={output}",
    ]
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, (
        f"torchgeo-bench exited {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    assert output.exists(), f"Expected results CSV at {output}"

    df = pd.read_csv(output)
    rows = df[df["dataset"] == "m-eurosat"]
    assert not rows.empty, f"No m-eurosat rows in {output}\n{df}"

    assert "bands" in df.columns, f"`bands` column missing from CSV: {df.columns.tolist()}"
    assert (rows["bands"] == "all").all(), (
        f"Expected bands=all for every m-eurosat row, got {rows['bands'].unique().tolist()}"
    )

    methods = set(rows["method"].unique())
    assert {"knn5", "linear"}.issubset(methods), (
        f"Expected knn5 and linear methods, got {sorted(methods)}"
    )

    feature_dims = rows["feature_dim"].unique().tolist()
    assert feature_dims == [512], f"Expected resnet18 feature_dim=512, got {feature_dims}"

    # Sanity-check accuracies are floats in [0, 1].
    metric_values = rows["metric_value"].astype(float)
    assert metric_values.between(0.0, 1.0).all(), (
        f"metric_value out of range: {metric_values.tolist()}"
    )
