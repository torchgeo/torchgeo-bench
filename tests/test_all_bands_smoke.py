"""Check all-band CSV results with real m-eurosat data.

Runs ``torchgeo-bench run model=timm/resnet18 dataset.bands=all`` on a small ``m-eurosat`` partition. Marked ``slow`` because it runs feature extraction on real data.
"""

from pathlib import Path

import pandas as pd
import pytest

from .test_cli_program import run_cli
from .test_integration import require_dataset_data


@pytest.mark.slow
def test_all_bands_e2e(tmp_path: Path):
    require_dataset_data("m-eurosat")

    output = tmp_path / "results.csv"
    cmd = [
        "run",
        "model=timm/resnet18",
        "model.pretrained=false",
        "model.seed=0",
        "dataset.names=[m-eurosat]",
        "dataset.bands=all",
        "dataset.partition=0.01x_train",
        "dataset.image_size=32",
        "dataset.batch_size=16",
        "dataset.num_workers=0",
        "eval.bootstrap=10",
        "eval.c_range=[-2,2,3]",
        "device=cpu",
        f"output={output}",
    ]
    completed = run_cli(*cmd, cwd=Path.cwd(), timeout=600)
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

    metric_values = rows["metric_value"].astype(float)
    assert metric_values.between(0.0, 1.0).all(), (
        f"metric_value out of range: {metric_values.tolist()}"
    )
