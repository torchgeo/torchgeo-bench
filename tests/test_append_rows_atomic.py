"""Persistence and locking contracts for the results CSV writer."""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pandas as pd
import pytest

from torchgeo_bench.results import append_rows_atomic


def _read_csv(path: str) -> list[list[str]]:
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_creates_file_with_header(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])

    rows = _read_csv(path)
    assert rows == [["a", "b"], ["1", "2"]]


def test_append_same_schema_does_not_duplicate_header(tmp_path: Path) -> None:
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


def test_schema_drift_added_column_rewrites_with_unioned_header(tmp_path: Path) -> None:
    """Appending a new column must not put its values outside the CSV header."""
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2}])
    append_rows_atomic(path, [{"a": 3, "b": 4, "c": "rgb"}])

    rows = _read_csv(path)
    assert rows[0] == ["a", "b", "c"], "header must include the new column"
    assert rows[1] == ["1", "2", ""], "legacy row gets empty value for new column"
    assert rows[2] == ["3", "4", "rgb"], "new row is fully populated"
    assert all(len(r) == len(rows[0]) for r in rows)


def test_schema_drift_removed_column_keeps_old_values(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": 1, "b": 2, "c": "rgb"}])
    append_rows_atomic(path, [{"a": 3, "b": 4}])

    rows = _read_csv(path)
    assert rows[0] == ["a", "b", "c"]
    assert rows[1] == ["1", "2", "rgb"]
    assert rows[2] == ["3", "4", ""]


@pytest.mark.parametrize("existing", [False, True])
def test_empty_rows_is_noop(tmp_path: Path, *, existing: bool) -> None:
    path = tmp_path / "out.csv"
    if existing:
        path.write_text("a,b\n1,2\n")
    append_rows_atomic(str(path), [])
    assert path.exists() is existing
    assert not path.with_suffix(".csv.lock").exists()
    if existing:
        assert path.read_text() == "a,b\n1,2\n"


def test_reordered_columns_keep_values_associated_with_names(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    append_rows_atomic(path, [{"a": "first", "b": "with,\nnewline"}])
    append_rows_atomic(path, [{"b": "second", "a": "last"}])
    assert pd.read_csv(path).to_dict("records") == [
        {"a": "first", "b": "with,\nnewline"},
        {"a": "last", "b": "second"},
    ]


def test_concurrent_writers_preserve_every_row_during_schema_drift(tmp_path: Path) -> None:
    path = str(tmp_path / "out.csv")
    barrier = Barrier(4)

    def write_batch(worker: int) -> None:
        barrier.wait(timeout=10)
        append_rows_atomic(
            path,
            [{"id": worker * 3 + i, f"worker_{worker}": f"value-{i}"} for i in range(3)],
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_batch, range(4)))

    rows = pd.read_csv(path).set_index("id")
    assert sorted(rows.index) == list(range(12))
    for worker in range(4):
        for index in range(3):
            row = rows.loc[worker * 3 + index]
            assert row[f"worker_{worker}"] == f"value-{index}"
            assert row.notna().sum() == 1


def test_read_failure_preserves_file_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "out.csv"
    append_rows_atomic(str(path), [{"a": 1}])
    before = path.read_bytes()

    def fail_read(*args: object, **kwargs: object) -> pd.DataFrame:
        raise pd.errors.ParserError("invalid CSV")

    with monkeypatch.context() as patch:
        patch.setattr(pd, "read_csv", fail_read)
        with pytest.raises(pd.errors.ParserError, match="invalid CSV"):
            append_rows_atomic(str(path), [{"a": 2}])
    assert path.read_bytes() == before
    append_rows_atomic(str(path), [{"a": 2}])
    assert _read_csv(str(path)) == [["a"], ["1"], ["2"]]
