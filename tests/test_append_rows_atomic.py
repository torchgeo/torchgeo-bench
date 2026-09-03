"""Tests for atomic result CSV writes."""

import csv
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from torchgeo_bench.main import _completed_run_keys, _profile_metric_names
from torchgeo_bench.results import ResultSchemaError, append_rows_atomic, load_results


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


def test_schema_drift_fails_without_rewriting_existing_file(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])
    original = (tmp_path / "out.csv").read_bytes()

    with pytest.raises(ResultSchemaError, match="schema mismatch"):
        append_rows_atomic(path, [{"a": 3, "b": 4, "c": "rgb"}])

    assert (tmp_path / "out.csv").read_bytes() == original


def test_rows_with_different_columns_are_rejected(tmp_path):
    path = str(tmp_path / "out.csv")

    with pytest.raises(ResultSchemaError, match="Row 1 columns"):
        append_rows_atomic(path, [{"a": 1}, {"a": 2, "b": 3}])

    assert not (tmp_path / "out.csv").exists()


def test_empty_rows_is_noop(tmp_path):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [])
    with pytest.raises(FileNotFoundError):
        open(path).close()


def test_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])
    original = (tmp_path / "out.csv").read_bytes()

    def fail_to_csv(self, file, *args, **kwargs):
        del self, args, kwargs
        file.write("partial")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)
    with pytest.raises(OSError, match="disk full"):
        append_rows_atomic(path, [{"a": 3, "b": 4}])

    assert (tmp_path / "out.csv").read_bytes() == original


def test_concurrent_writers_preserve_every_row(tmp_path):
    path = str(tmp_path / "out.csv")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: append_rows_atomic(path, [{"value": value}]), range(20)))

    frame = pd.read_csv(path)
    assert sorted(frame["value"]) == list(range(20))


def test_load_results_reports_malformed_file_path(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_bytes(b"\xff")

    with pytest.raises(ResultSchemaError, match=re.escape(str(path))):
        load_results(tmp_path)


def test_load_results_rejects_mixed_schemas(tmp_path):
    pd.DataFrame([{"a": 1}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"a": 2, "b": 3}]).to_csv(tmp_path / "b.csv", index=False)

    with pytest.raises(ResultSchemaError, match="schema mismatch"):
        load_results(tmp_path)


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
