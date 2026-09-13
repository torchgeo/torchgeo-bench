"""Numeric and string-only probability artifact loading."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from projects.cleanlab import (
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
def test_consumers_reject_object_arrays(tmp_path: Path, consumer: str, member: str) -> None:
    path = tmp_path / "example__model_test.npz"
    arrays = _arrays()
    arrays[member] = np.array([_ObjectValue()], dtype=object)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="Object arrays"):
        _read(consumer, path, tmp_path / "out")


@pytest.mark.parametrize("consumer", ["single", "multi", "audit"])
def test_consumers_reject_pickle_disguised_as_npz(tmp_path: Path, consumer: str) -> None:
    path = tmp_path / "example__model_test.npz"
    path.write_bytes(pickle.dumps(_ObjectValue()))
    with pytest.raises(ValueError, match="pickl"):
        _read(consumer, path, tmp_path / "out")


@pytest.mark.parametrize("task", ["single", "multi"])
def test_audit_reads_numeric_and_unicode_artifacts(tmp_path: Path, task: str) -> None:
    path = tmp_path / "example__model_test.npz"
    arrays = _arrays()
    if task == "multi":
        arrays["labels"] = np.eye(2, dtype=np.int64)
    np.savez(
        path,
        **arrays,
        meta=np.array(["example", "model", "test"], dtype=str),
        sample_ids=np.array(["tile_北", "tile_南"], dtype=str),
    )
    result = run_cleanlab_audit._load_npz(path)
    assert set(result) == {"labels", "probs", "classes"}
    for name, expected in arrays.items():
        np.testing.assert_array_equal(result[name], expected)


def test_multilabel_report_reads_numeric_and_unicode_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "example__model_test.npz"
    arrays = _arrays()
    arrays["labels"] = np.eye(2, dtype=np.int64)
    np.savez(
        path,
        **arrays,
        meta=np.array(["example", "model"], dtype=str),
        sample_ids=np.array(["tile_北", "tile_南"], dtype=str),
    )
    monkeypatch.setattr(
        cleanlab_per_class_multilabel,
        "_per_class_flags",
        lambda y, probs: np.zeros_like(y, dtype=bool),
    )
    report = cleanlab_per_class_multilabel.report_dataset(path, tmp_path / "out")
    np.testing.assert_array_equal(report["n_pos"], [1, 1])
    np.testing.assert_array_equal(report["ap"], [1.0, 1.0])
    np.testing.assert_array_equal(report["n_flagged"], [0, 0])
    assert (tmp_path / "out" / "perclass_example_test.csv").is_file()


@pytest.mark.usefixtures("cleanlab_filter")
def test_singlelabel_report_reads_numeric_and_unicode_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "example__model_test.npz"
    labels = np.tile([0, 1], 10)
    np.savez(
        path,
        labels=labels,
        probs=0.8 * np.eye(2)[labels] + 0.1,
        classes=np.array([0, 1]),
        meta=np.array(["example", "model"], dtype=str),
        sample_ids=np.array([f"tile_{i}" for i in range(len(labels))], dtype=str),
    )
    report = cleanlab_per_class_singlelabel.report_dataset(path, tmp_path / "out")
    np.testing.assert_array_equal(report["n"], [10, 10])
    np.testing.assert_array_equal(report["acc"], [1.0, 1.0])
    np.testing.assert_array_equal(report["n_flagged"], [0, 0])
    assert (tmp_path / "out" / "perclass_example_test.csv").is_file()


def test_audit_reads_numeric_members_without_loading_unused_objects(tmp_path: Path) -> None:
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


def test_audit_model_name_comes_from_the_artifact_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    def audit(labels: np.ndarray, probs: np.ndarray, classes: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {"given_label": labels, "guessed_label": labels, "is_issue": [False, False]}
        )

    monkeypatch.setattr(run_cleanlab_audit, "_audit_singlelabel", audit)
    run_cleanlab_audit.main()
    summary = pd.read_csv(output / "summary.csv")
    assert summary.iloc[0]["model"] == "model_with_underscores"
    assert summary.iloc[0]["n"] == 2


@pytest.mark.parametrize("dataset", ["m-eurosat", "m-bigearthnet"])
def test_probability_writer_emits_no_object_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dataset: str
) -> None:
    import torch
    from torch.utils.data import DataLoader

    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset": dataset,
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
            dataset,
            "--results",
            str(results),
            "--out",
            str(output),
            "--device",
            "cpu",
        ],
    )
    train_labels = np.array([0, 1, 0, 1], dtype=np.int64)
    val_labels = np.array([1, 0], dtype=np.int64)
    test_labels = np.array([1, 0, 1], dtype=np.int64)
    if dataset == "m-bigearthnet":
        train_labels, val_labels, test_labels = [
            np.eye(2, dtype=np.float32)[labels]
            for labels in (train_labels, val_labels, test_labels)
        ]

    def samples(labels: np.ndarray) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "image": torch.full((3, 2, 2), float(index)),
                "label": torch.from_numpy(np.asarray(label)),
            }
            for index, label in enumerate(labels)
        ]

    train = samples(train_labels)
    train_loader = DataLoader(
        train, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(17)
    )
    val_loader = DataLoader(samples(val_labels), batch_size=2)
    test_loader = DataLoader(samples(test_labels), batch_size=2)
    monkeypatch.setattr(
        cleanlab_extract_probs,
        "get_datasets",
        lambda **kwargs: (train, train_loader, val_loader, test_loader),
    )
    monkeypatch.setattr(
        cleanlab_extract_probs, "instantiate", lambda *args, **kwargs: torch.nn.Identity()
    )

    def embed(
        model: torch.nn.Module, loader: DataLoader, device: torch.device, **kwargs: object
    ) -> tuple[np.ndarray, np.ndarray]:
        labels = np.concatenate([batch["label"].numpy() for batch in loader])
        return np.zeros((len(labels), 2), dtype=np.float32), labels

    monkeypatch.setattr(cleanlab_extract_probs, "embed_split", embed)

    class Probe:
        def __init__(self, **kwargs: object) -> None:
            self.classes_ = np.array([0, 1], dtype=np.int64)

        def fit(self, images: torch.Tensor, labels: torch.Tensor) -> None:
            np.testing.assert_array_equal(
                labels.numpy(), np.concatenate([train_labels, val_labels])
            )
            assert len(images) == len(train_labels) + len(val_labels)

        def predict_proba(self, images: torch.Tensor) -> np.ndarray:
            return np.tile(np.array([[0.75, 0.25]], dtype=np.float32), (len(images), 1))

    monkeypatch.setattr(cleanlab_extract_probs, "LogisticRegression", Probe)
    with torch.random.fork_rng(devices=[]):
        cleanlab_extract_probs.main()
    for split, labels in (("train", train_labels), ("test", test_labels)):
        path = output / f"{dataset}__rcf_{split}.npz"
        with np.load(path, allow_pickle=False) as archive:
            assert all(not archive[name].dtype.hasobject for name in archive.files)
            assert archive["meta"].dtype.kind == "U"
            assert archive["meta"][1] == "rcf"
            assert archive["meta"][-1] == split
            np.testing.assert_array_equal(archive["indices"], np.arange(len(labels)))
            np.testing.assert_array_equal(archive["labels"], labels)
            np.testing.assert_array_equal(
                archive["probs"],
                np.tile(np.array([[0.75, 0.25]], dtype=np.float32), (len(labels), 1)),
            )
        np.testing.assert_array_equal(run_cleanlab_audit._load_npz(path)["labels"], labels)
