"""Numeric and string-only probability artifact loading."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from experiments.scripts import (
    cleanlab_extract_probs,
    cleanlab_per_class_multilabel,
    cleanlab_per_class_singlelabel,
    run_cleanlab_audit,
)

EXECUTED: list[bool] = []


def _record_execution() -> int:
    EXECUTED.append(True)
    return 0


class _ObjectValue:
    def __reduce__(self) -> tuple:
        return _record_execution, ()


@pytest.fixture(autouse=True)
def _no_object_execution() -> Iterator[None]:
    EXECUTED.clear()
    yield
    assert not EXECUTED


def _arrays() -> dict[str, np.ndarray]:
    return {
        "labels": np.array([0, 1], dtype=np.int64),
        "probs": np.array([[0.9, 0.1], [0.2, 0.8]], dtype=np.float32),
        "classes": np.array([0, 1], dtype=np.int64),
    }


def _read(consumer: str, path: Path, out: Path) -> None:
    if consumer == "single":
        cleanlab_per_class_singlelabel.report_dataset(path, out)
    elif consumer == "multi":
        cleanlab_per_class_multilabel.report_dataset(path, out)
    else:
        run_cleanlab_audit._load_npz(path)


@pytest.mark.parametrize(
    ("consumer", "member"),
    [
        ("single", "labels"),
        ("single", "probs"),
        ("single", "classes"),
        ("multi", "labels"),
        ("multi", "probs"),
        ("audit", "labels"),
        ("audit", "probs"),
        ("audit", "classes"),
    ],
)
def test_consumers_reject_object_arrays(tmp_path, consumer, member) -> None:
    path = tmp_path / "example__model_test.npz"
    arrays = _arrays()
    arrays[member] = np.array([_ObjectValue()], dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Object arrays"):
        _read(consumer, path, tmp_path / "out")


@pytest.mark.parametrize("consumer", ["single", "multi", "audit"])
def test_consumers_reject_pickle_disguised_as_npz(tmp_path, consumer) -> None:
    path = tmp_path / "example__model_test.npz"
    path.write_bytes(pickle.dumps(_ObjectValue()))
    with pytest.raises(ValueError):
        _read(consumer, path, tmp_path / "out")


def test_audit_reads_numeric_members_without_loading_unused_objects(tmp_path) -> None:
    path = tmp_path / "example__model_test.npz"
    arrays = _arrays()
    np.savez(
        path,
        **arrays,
        meta=np.array([_ObjectValue()], dtype=object),
        unused=np.array([_ObjectValue()], dtype=object),
    )
    result = run_cleanlab_audit._load_npz(path)
    for name, expected in arrays.items():
        np.testing.assert_array_equal(result[name], expected)


def test_audit_model_name_comes_from_the_artifact_name(tmp_path, monkeypatch) -> None:
    source = tmp_path / "probs"
    source.mkdir()
    np.savez(
        source / "example__model_with_underscores_test.npz",
        **_arrays(),
        meta=np.array([_ObjectValue()], dtype=object),
    )
    output = tmp_path / "audit"
    monkeypatch.setattr(
        "sys.argv", ["run_cleanlab_audit.py", "--probs-dir", str(source), "--out-dir", str(output)]
    )

    def audit(labels, probs, classes) -> pd.DataFrame:
        return pd.DataFrame(
            {"given_label": labels, "guessed_label": labels, "is_issue": [False, False]}
        )

    monkeypatch.setattr(run_cleanlab_audit, "_audit_singlelabel", audit)
    run_cleanlab_audit.main()
    summary = pd.read_csv(output / "summary.csv")
    assert summary.iloc[0]["model"] == "model_with_underscores"
    assert summary.iloc[0]["n"] == 2


def test_probability_writer_emits_no_object_arrays(tmp_path, monkeypatch) -> None:
    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset": "m-eurosat",
                "method": "linear",
                "name": "rcf",
                "metric_value": 0.9,
                "normalization": "bandspec_zscore",
                "bands": "rgb",
                "image_size": 16,
                "interpolation": "bilinear",
                "partition": "default",
                "best_c": 1.0,
            }
        ]
    ).to_csv(results, index=False)
    output = tmp_path / "probs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cleanlab_extract_probs.py",
            "--dataset",
            "m-eurosat",
            "--results",
            str(results),
            "--out",
            str(output),
            "--device",
            "cpu",
        ],
    )
    samples = [{"image": torch.zeros(3, 2, 2), "label": torch.tensor(i % 2)} for i in range(4)]
    loader = DataLoader(samples, batch_size=2)
    monkeypatch.setattr(
        cleanlab_extract_probs, "get_datasets", lambda **kwargs: (samples, loader, loader, loader)
    )
    monkeypatch.setattr(
        cleanlab_extract_probs, "instantiate", lambda *args, **kwargs: torch.nn.Identity()
    )
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    monkeypatch.setattr(
        cleanlab_extract_probs,
        "embed_split",
        lambda *args, **kwargs: (np.zeros((4, 2), dtype=np.float32), labels),
    )

    class Probe:
        def __init__(self, **kwargs) -> None:
            self.classes_ = np.array([0, 1], dtype=np.int64)

        def fit(self, images, labels) -> None:
            pass

        def predict_proba(self, images) -> np.ndarray:
            return np.tile(np.array([[0.75, 0.25]], dtype=np.float32), (len(images), 1))

    monkeypatch.setattr(cleanlab_extract_probs, "LogisticRegression", Probe)
    cleanlab_extract_probs.main()
    for split in ("train", "test"):
        path = output / f"m-eurosat__rcf_{split}.npz"
        with np.load(path, allow_pickle=False) as archive:
            assert all(not archive[name].dtype.hasobject for name in archive.files)
            assert archive["meta"][1] == "rcf"
            assert archive["meta"][-1] == split
            np.testing.assert_array_equal(archive["labels"], labels)
        np.testing.assert_array_equal(run_cleanlab_audit._load_npz(path)["labels"], labels)
