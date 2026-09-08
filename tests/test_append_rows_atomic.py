"""Tests for ``torchgeo_bench.main.append_rows_atomic`` (CSV writer)."""

import csv

import pandas as pd
import pytest

from torchgeo_bench.main import _completed_run_keys, _profile_metric_names, append_rows_atomic


def _read_csv(path: str) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_creates_file_with_header(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])

    rows = _read_csv(path)
    assert rows == [["a", "b"], ["1", "2"]]


def test_append_same_schema_does_not_duplicate_header(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])
    append_rows_atomic(path, [{"a": 3, "b": 4}, {"a": 5, "b": 6}])

    rows = _read_csv(path)
    assert rows == [
        ["a", "b"],
        ["1", "2"],
        ["3", "4"],
        ["5", "6"],
    ]


def test_schema_drift_added_column_rewrites_with_unioned_header(tmp_path):
    """Appending a new column must not put its values outside the CSV header."""
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])
    append_rows_atomic(path, [{"a": 3, "b": 4, "c": "rgb"}])

    rows = _read_csv(path)
    assert rows[0] == ["a", "b", "c"], "header must include the new column"
    assert rows[1] == ["1", "2", ""], "legacy row gets empty value for new column"
    assert rows[2] == ["3", "4", "rgb"], "new row is fully populated"
    assert all(len(r) == len(rows[0]) for r in rows)


def test_schema_drift_removed_column_keeps_old_values(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2, "c": "rgb"}])
    append_rows_atomic(path, [{"a": 3, "b": 4}])

    rows = _read_csv(path)
    assert rows[0] == ["a", "b", "c"]
    assert rows[1] == ["1", "2", "rgb"]
    assert rows[2] == ["3", "4", ""]


def test_empty_rows_is_noop(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [])
    with pytest.raises(FileNotFoundError):
        open(path).close()


def test_resume_keys_can_require_metric_name():
    key_cols = ("dataset", "method", "model", "name")
    df = pd.DataFrame(
        [
            {
                "dataset": "m-eurosat",
                "method": "intrinsic_dim",
                "model": "M",
                "name": "n",
                "metric_name": "id_twonn_train",
            }
        ]
    )
    key = ("m-eurosat", "intrinsic_dim", "M", "n")
    assert key in _completed_run_keys(df, key_cols, "id_twonn_train")
    assert key not in _completed_run_keys(df, key_cols, "id_mle_train")


def test_profile_resume_requires_multiple_metrics():
    metrics = _profile_metric_names(None)
    assert "throughput_samples_per_sec" in metrics
    assert "latency_ms_per_batch_p50" in metrics
    assert "params_m" in metrics
