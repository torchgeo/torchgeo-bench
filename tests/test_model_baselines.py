"""Tests for model accuracy baselines."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "accuracy_baselines.csv"
_RESULTS_DIR = _REPO_ROOT / "results" / "models"

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

_V1_DATA = Path("data/classification_v1.0")
_V2_DATA = Path("data/geobenchv2")

_V1_DATASETS = {"m-eurosat", "m-forestnet", "m-so2sat", "m-pv4ger", "m-brick-kiln", "m-bigearthnet"}


def _dataset_data_exists(dataset: str) -> bool:
    if dataset in _V1_DATASETS:
        return _V1_DATA.exists()
    return (_V2_DATA / dataset).exists()


def _run_bench(*overrides: str, timeout: int = 600) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "torchgeo_bench", *overrides]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(_REPO_ROOT),
    )


@pytest.mark.skipif(
    not list(_RESULTS_DIR.glob("*.csv")), reason="no per-model results in results/models"
)
def test_update_baselines_script_runs(tmp_path: Path) -> None:
    """Script runs, exits 0, and outputs a CSV with expected columns."""
    out = tmp_path / "out.csv"
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "update_baselines.py"), "--output", str(out)],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert out.exists()
    df = pd.read_csv(out)
    assert _FIXTURE_COLS.issubset(set(df.columns))


# Load fixture at module level for parametrisation (empty DF if file absent)
_fixture_df: pd.DataFrame
if _FIXTURE_PATH.exists():
    _fixture_df = pd.read_csv(_FIXTURE_PATH)
else:
    _fixture_df = pd.DataFrame(columns=list(_FIXTURE_COLS))

_SENTINELS = {
    ("rcf", "m-eurosat", "rgb"),
    ("timm/resnet18", "m-eurosat", "rgb"),
    ("torchgeo/dofa_base", "so2sat", "all"),
    ("imagestats", "m-pv4ger", "all"),
}
_COMBOS = [
    combo
    for combo in _fixture_df[["model_config", "name", "dataset", "bands"]]
    .drop_duplicates()
    .to_dict("records")
    if (combo["model_config"], combo["dataset"], combo["bands"]) in _SENTINELS
]


def _combo_id(combo: dict) -> str:
    config = combo["model_config"].replace("/", "_")
    return f"{config}__{combo['dataset']}__{combo['bands']}"


@pytest.mark.accuracy_check
@pytest.mark.parametrize("combo", _COMBOS, ids=[_combo_id(c) for c in _COMBOS])
def test_accuracy(combo: dict, tmp_path: Path) -> None:
    """Run bench CLI for one (model_config, dataset, bands) combo and check accuracy."""
    model_config = combo["model_config"]
    dataset = combo["dataset"]
    bands = combo["bands"]

    if not _dataset_data_exists(dataset):
        pytest.skip(f"Dataset data not found for {dataset}")

    out = tmp_path / "out.csv"
    result = _run_bench(
        "run",
        "--model",
        model_config,
        "--datasets",
        dataset,
        "--bands",
        bands,
        "--output",
        str(out),
        "--bootstrap",
        "10",
        "--device",
        _DEVICE,
    )
    assert result.returncode == 0, f"CLI failed for {model_config} × {dataset}:\n{result.stderr}"

    actual_df = pd.read_csv(out)
    fixture_rows = _fixture_df[
        (_fixture_df["model_config"] == model_config)
        & (_fixture_df["dataset"] == dataset)
        & (_fixture_df["bands"] == bands)
    ]
    for _, row in fixture_rows.iterrows():
        method = row["method"]
        match = actual_df[actual_df["method"] == method]
        assert len(match) > 0, f"Method {method} not found in output for {model_config} × {dataset}"
        actual = match.iloc[0]["metric_value"]
        expected = row["expected_value"]
        assert actual == pytest.approx(expected, abs=_TOL), (
            f"{model_config} × {dataset} {method}: got {actual:.4f}, expected {expected:.4f} ±{_TOL}"
        )
