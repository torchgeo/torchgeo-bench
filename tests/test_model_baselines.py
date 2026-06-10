"""Tests for model accuracy baselines fixture pipeline and regression checks."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "accuracy_baselines.csv"
_ALL_RESULTS = _REPO_ROOT / "results" / "all_results.csv"

_EXPECTED_FIXTURE_COLS = {
    "model_config",
    "name",
    "dataset",
    "method",
    "metric_name",
    "bands",
    "partition",
    "expected_value",
}


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


# ---------------------------------------------------------------------------
# Slice 2: filter_and_deduplicate unit tests (offline, no accuracy_check mark)
# ---------------------------------------------------------------------------


def test_filter_and_deduplicate_picks_canonical_bands() -> None:
    """Deduplication picks the canonical bands row, dropping others."""
    from scripts.update_baselines import filter_and_deduplicate

    df = pd.DataFrame(
        {
            "name": ["olmoearth_v1_nano", "olmoearth_v1_nano"],
            "dataset": ["m-eurosat", "m-eurosat"],
            "method": ["knn5", "knn5"],
            "metric_name": ["accuracy", "accuracy"],
            "bands": ["rgb", "all"],
            "partition": ["default", "default"],
            "metric_value": [0.80, 0.85],
            "model": ["torchgeo_bench.models.OlmoEarthBenchModel"] * 2,
        }
    )
    canonical = {"olmoearth_v1_nano": "all"}
    result = filter_and_deduplicate(
        df,
        canonical_bands=canonical,
        pinned_names={"olmoearth_v1_nano"},
        target_datasets={"m-eurosat"},
    )
    assert len(result) == 1
    assert result.iloc[0]["bands"] == "all"


def test_filter_and_deduplicate_drops_non_pinned_models() -> None:
    """Rows for models not in the pinned set are dropped."""
    from scripts.update_baselines import filter_and_deduplicate

    df = pd.DataFrame(
        {
            "name": ["olmoearth_v1_nano", "some_other_model"],
            "dataset": ["m-eurosat", "m-eurosat"],
            "method": ["knn5", "knn5"],
            "metric_name": ["accuracy", "accuracy"],
            "bands": ["all", "all"],
            "partition": ["default", "default"],
            "metric_value": [0.85, 0.50],
            "model": [
                "torchgeo_bench.models.OlmoEarthBenchModel",
                "torchgeo_bench.models.SomeBench",
            ],
        }
    )
    canonical = {"olmoearth_v1_nano": "all"}
    result = filter_and_deduplicate(
        df,
        canonical_bands=canonical,
        pinned_names={"olmoearth_v1_nano"},
        target_datasets={"m-eurosat"},
    )
    assert set(result["name"].unique()) == {"olmoearth_v1_nano"}
    assert len(result) == 1


def test_filter_and_deduplicate_output_schema() -> None:
    """Output DataFrame has exactly the expected columns."""
    from scripts.update_baselines import filter_and_deduplicate

    df = pd.DataFrame(
        {
            "name": ["rcf"],
            "dataset": ["m-eurosat"],
            "method": ["knn5"],
            "metric_name": ["accuracy"],
            "bands": ["all"],
            "partition": ["default"],
            "metric_value": [0.61],
            "model": ["torchgeo_bench.models.RCFBench"],
        }
    )
    canonical = {"rcf": "all"}
    result = filter_and_deduplicate(
        df,
        canonical_bands=canonical,
        pinned_names={"rcf"},
        target_datasets={"m-eurosat"},
    )
    assert set(result.columns) == _EXPECTED_FIXTURE_COLS


def test_filter_and_deduplicate_missing_combo_produces_no_row() -> None:
    """Missing model × dataset combos produce no row (not an error)."""
    from scripts.update_baselines import filter_and_deduplicate

    df = pd.DataFrame(
        {
            "name": ["rcf"],
            "dataset": ["m-eurosat"],
            "method": ["knn5"],
            "metric_name": ["accuracy"],
            "bands": ["all"],
            "partition": ["default"],
            "metric_value": [0.61],
            "model": ["torchgeo_bench.models.RCFBench"],
        }
    )
    canonical = {"rcf": "all", "tgeo_croma_base": "all"}
    result = filter_and_deduplicate(
        df,
        canonical_bands=canonical,
        pinned_names={"rcf", "tgeo_croma_base"},
        target_datasets={"m-eurosat", "benv2"},
    )
    # tgeo_croma_base × benv2 is missing — no row, no error
    assert not result[(result["name"] == "tgeo_croma_base") & (result["dataset"] == "benv2")].shape[0]


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
    assert out.exists(), "Output CSV was not created"
    df = pd.read_csv(out)
    assert _EXPECTED_FIXTURE_COLS.issubset(set(df.columns))


# ---------------------------------------------------------------------------
# Slice 3: offline fixture validation tests (no accuracy_check mark)
# ---------------------------------------------------------------------------


def test_fixture_loads_and_has_expected_columns() -> None:
    """Fixture CSV exists, is non-empty, and has the required columns."""
    assert _FIXTURE_PATH.exists(), f"Fixture not found at {_FIXTURE_PATH}"
    df = pd.read_csv(_FIXTURE_PATH)
    assert not df.empty
    assert _EXPECTED_FIXTURE_COLS.issubset(set(df.columns))


def test_parametrize_ids_are_unique() -> None:
    """Derived pytest IDs from fixture combos are unique."""
    assert _FIXTURE_PATH.exists(), f"Fixture not found at {_FIXTURE_PATH}"
    df = pd.read_csv(_FIXTURE_PATH)
    combos = df[["model_config", "dataset", "bands"]].drop_duplicates()
    ids = [
        f"{row['model_config'].replace('/', '_')}_{row['dataset']}_{row['bands']}"
        for _, row in combos.iterrows()
    ]
    assert len(ids) == len(set(ids)), f"Duplicate pytest IDs found: {ids}"


# ---------------------------------------------------------------------------
# Slice 3: parametrised accuracy_check tests
# ---------------------------------------------------------------------------

_TOL = 0.02

_V1_DATA = Path("data/classification_v1.0")
_V2_DATA = Path("data/geobenchv2")

_V1_DATASETS = {"m-eurosat", "m-forestnet", "m-so2sat", "m-pv4ger", "m-brick-kiln", "m-bigearthnet"}
_V2_DATASETS = {"benv2", "treesatai", "so2sat", "forestnet"}


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


# Load fixture at module level for parametrisation (empty DF if file absent)
_fixture_df: pd.DataFrame
if _FIXTURE_PATH.exists():
    _fixture_df = pd.read_csv(_FIXTURE_PATH)
else:
    _fixture_df = pd.DataFrame(columns=list(_EXPECTED_FIXTURE_COLS))

_COMBOS = (
    _fixture_df[["model_config", "name", "dataset", "bands"]]
    .drop_duplicates()
    .to_dict("records")
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
