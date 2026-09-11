"""Tests for model accuracy baselines."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.support.cli import run_cli
from tests.support.data import require_dataset_data

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "accuracy_baselines.csv"

_FIXTURE_COLS = {
    "model_config",
    "name",
    "dataset",
    "method",
    "metric_name",
    "bands",
    "partition",
    "expected_value",
}

_TOL = 0.02


def test_update_baselines_script_filters_and_deduplicates(tmp_path: Path) -> None:
    row = {
        "name": "rcf",
        "dataset": "m-eurosat",
        "method": "knn5",
        "metric_name": "accuracy",
        "bands": "all",
        "partition": "default",
        "metric_value": 0.75,
    }
    source = tmp_path / "results.csv"
    pd.DataFrame(
        [
            row,
            {**row, "metric_value": 0.80},
            {**row, "method": "linear", "metric_value": 0.85},
            {**row, "bands": "rgb"},
            {**row, "metric_name": "micro_mAP"},
            {**row, "partition": "tiny"},
            {**row, "name": "unregistered-model"},
            {**row, "dataset": "unselected-dataset"},
            {**row, "method": "seg-linear"},
        ]
    ).to_csv(source, index=False)
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "update_baselines.py"),
            "--input",
            str(source),
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    df = pd.read_csv(out)
    assert set(df.columns) == _FIXTURE_COLS
    assert df.to_dict("records") == [
        {
            "model_config": "rcf",
            "name": "rcf",
            "dataset": "m-eurosat",
            "method": method,
            "metric_name": "accuracy",
            "bands": "all",
            "partition": "default",
            "expected_value": expected,
        }
        for method, expected in [("knn5", 0.75), ("linear", 0.85)]
    ]


_fixture_df = pd.read_csv(_FIXTURE_PATH)

_COMBOS = (
    _fixture_df[["model_config", "name", "dataset", "bands"]].drop_duplicates().to_dict("records")
)


def _combo_id(combo: dict[str, str]) -> str:
    config = combo["model_config"].replace("/", "_")
    return f"{config}__{combo['dataset']}__{combo['bands']}"


@pytest.mark.accuracy_check
@pytest.mark.parametrize("combo", _COMBOS, ids=[_combo_id(c) for c in _COMBOS])
def test_accuracy(combo: dict[str, str], tmp_path: Path) -> None:
    model_config = combo["model_config"]
    dataset = combo["dataset"]
    bands = combo["bands"]

    require_dataset_data(dataset)

    out = tmp_path / "out.csv"
    result = run_cli(
        "run",
        f"model={model_config}",
        f"dataset.names=[{dataset}]",
        f"dataset.bands={bands}",
        f"output={out}",
        "eval.bootstrap=10",
        "device=cpu",
        cwd=Path.cwd(),
        timeout=600,
        offline=False,
    )
    assert result.returncode == 0, f"CLI failed for {model_config} x {dataset}:\n{result.stderr}"

    actual_df = pd.read_csv(out)
    fixture_rows = _fixture_df[
        (_fixture_df["model_config"] == model_config)
        & (_fixture_df["name"] == combo["name"])
        & (_fixture_df["dataset"] == dataset)
        & (_fixture_df["bands"] == bands)
    ]
    for _, row in fixture_rows.iterrows():
        method = row["method"]
        match = actual_df[
            (actual_df["method"] == method)
            & (actual_df["name"] == row["name"])
            & (actual_df["metric_name"] == row["metric_name"])
            & (actual_df["partition"] == row["partition"])
        ]
        assert len(match) == 1, f"Expected one {method} result for {model_config} x {dataset}"
        actual = match.iloc[0]["metric_value"]
        expected = row["expected_value"]
        assert actual == pytest.approx(expected, abs=_TOL), (
            f"{model_config} x {dataset} {method}: got {actual:.4f}, expected {expected:.4f} ±{_TOL}"
        )
