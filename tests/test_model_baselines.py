"""Tests for model accuracy baselines."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "accuracy_baselines.csv"
_ALL_RESULTS = _REPO_ROOT / "results" / "all_results.csv"

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


def test_accuracy_check_marker_is_registered() -> None:
    """Verify accuracy_check marker is listed in pytest --markers output."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0
    assert "accuracy_check" in result.stdout, (
        f"accuracy_check marker not registered; got markers:\n{result.stdout}"
    )


@pytest.mark.skipif(not _ALL_RESULTS.exists(), reason="results/all_results.csv not found")
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


def test_fixture_has_expected_columns() -> None:
    """Fixture CSV exists, is non-empty, and has the required columns."""
    assert _FIXTURE_PATH.exists(), f"Fixture not found at {_FIXTURE_PATH}"
    df = pd.read_csv(_FIXTURE_PATH)
    assert not df.empty
    assert _FIXTURE_COLS.issubset(set(df.columns))


def test_parametrize_ids_are_unique() -> None:
    """Derived pytest IDs from fixture combos are unique."""
    assert _FIXTURE_PATH.exists(), f"Fixture not found at {_FIXTURE_PATH}"
    df = pd.read_csv(_FIXTURE_PATH)
    combos = df[["model_config", "dataset", "bands"]].drop_duplicates()
    ids = [
        f"{row['model_config'].replace('/', '_')}__{row['dataset']}__{row['bands']}"
        for _, row in combos.iterrows()
    ]
    assert len(ids) == len(set(ids)), f"Duplicate pytest IDs: {ids}"


# Load fixture at module level for parametrisation (empty DF if file absent)
_fixture_df: pd.DataFrame
if _FIXTURE_PATH.exists():
    _fixture_df = pd.read_csv(_FIXTURE_PATH)
else:
    _fixture_df = pd.DataFrame(columns=list(_FIXTURE_COLS))

_COMBOS = (
    _fixture_df[["model_config", "name", "dataset", "bands"]].drop_duplicates().to_dict("records")
)


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
        f"model={model_config}",
        f"dataset.names=[{dataset}]",
        f"dataset.bands={bands}",
        f"output={out}",
        "eval.bootstrap=10",
        "device=cuda:0",
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
