"""Tests for read-time results de-duplication and OlmoEarth relabelling."""

import pandas as pd

from torchgeo_bench.results import (
    DEDUP_KEY_COLS,
    _dedup_results,
    _relabel_olmoearth_normalization,
    load_results,
)


def _row(**overrides) -> dict:
    row = dict.fromkeys(DEDUP_KEY_COLS, "x")
    row.update(
        {
            "dataset": "benv2",
            "method": "knn5",
            "metric_name": "micro_mAP",
            "model": "torchgeo_bench.models.TorchGeoDOFABench",
            "name": "tgeo_dofa_large",
            "normalization": "bandspec_zscore",
            "image_size": 224,
            "interpolation": "bilinear",
            "partition": "default",
            "bands": "rgb",
            "num_classes": 19,
            "config_hash": "",
            "metric_value": 0.5,
        }
    )
    row.update(overrides)
    return row


def test_hashed_row_wins_over_legacy_unhashed_row() -> None:
    df = pd.DataFrame([_row(metric_value=0.1), _row(config_hash="abc", metric_value=0.2)])
    out = _dedup_results(df)
    assert len(out) == 1
    assert out.iloc[0]["metric_value"] == 0.2


def test_last_appended_wins_among_hashed_rows() -> None:
    """A rerun under a perf-only config change appends a second hashed row."""
    df = pd.DataFrame(
        [
            _row(config_hash="aaa", metric_value=0.679566),
            _row(config_hash="bbb", metric_value=0.679501),
        ]
    )
    out = _dedup_results(df)
    assert len(out) == 1
    assert out.iloc[0]["metric_value"] == 0.679501


def test_distinct_measurements_are_kept() -> None:
    df = pd.DataFrame([_row(), _row(method="linear"), _row(bands="all")])
    assert len(_dedup_results(df)) == 3


def test_different_seeds_are_distinct_measurements() -> None:
    """Multi-seed sweeps must not collapse to a single arbitrary seed."""
    df = pd.DataFrame(
        [
            _row(seed=0, config_hash="a", metric_value=0.81),
            _row(seed=1, config_hash="b", metric_value=0.83),
        ]
    )
    out = _dedup_results(df)
    assert len(out) == 2
    assert sorted(out["seed"]) == [0, 1]


def test_metric_name_separates_rows_from_one_run() -> None:
    """One run emits several metrics under the same resume key; keep them all."""
    df = pd.DataFrame(
        [_row(config_hash="a", metric_name=m) for m in ("id_TwoNN_train", "id_MLE_train")]
    )
    assert len(_dedup_results(df)) == 2


def test_olmoearth_zscore_rows_are_relabelled_model_native() -> None:
    df = pd.DataFrame([_row(name="olmoearth_v1_2_base", normalization="bandspec_zscore")])
    assert _relabel_olmoearth_normalization(df).iloc[0]["normalization"] == "model_native"


def test_non_olmoearth_rows_keep_their_normalization() -> None:
    df = pd.DataFrame([_row(name="tgeo_dofa_large", normalization="bandspec_zscore")])
    assert _relabel_olmoearth_normalization(df).iloc[0]["normalization"] == "bandspec_zscore"


def test_relabel_collapses_olmoearth_duplicate_normalization_runs() -> None:
    """OlmoEarth ignores dataset.normalization, so the two runs are one measurement."""
    df = pd.DataFrame(
        [
            _row(name="olmoearth_v1_2_base", normalization="bandspec_zscore", config_hash="a"),
            _row(name="olmoearth_v1_2_base", normalization="model_native", config_hash="b"),
        ]
    )
    assert len(_dedup_results(_relabel_olmoearth_normalization(df))) == 1


def test_transforms_are_noop_on_frames_missing_columns() -> None:
    df = pd.DataFrame([{"dataset": "benv2", "metric_value": 1.0}])
    assert len(_dedup_results(df)) == 1
    assert len(_relabel_olmoearth_normalization(df)) == 1


def test_load_results_applies_both_transforms(tmp_path) -> None:
    pd.DataFrame(
        [
            _row(name="olmoearth_v1_2_base", normalization="bandspec_zscore", config_hash="a"),
            _row(name="olmoearth_v1_2_base", normalization="model_native", config_hash="b"),
        ]
    ).to_csv(tmp_path / "olmoearth_v1_2_base.csv", index=False)
    out = load_results(tmp_path)
    assert len(out) == 1
    assert out.iloc[0]["normalization"] == "model_native"
