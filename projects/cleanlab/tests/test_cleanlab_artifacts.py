"""Numeric and string-only probability artifact loading."""

import pickle
from collections.abc import Iterator
from functools import partial
from pathlib import Path

import cleanlab_per_class_multilabel
import cleanlab_per_class_singlelabel
import numpy as np
import pandas as pd
import pytest
import run_cleanlab_audit

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


def test_singlelabel_report_reads_numeric_and_unicode_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cleanlab")
    from cleanlab import filter as cleanlab_filter

    monkeypatch.setattr(
        cleanlab_filter,
        "find_label_issues",
        partial(cleanlab_filter.find_label_issues, n_jobs=1),
    )
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
@pytest.mark.parametrize("custom_model", [False, True])
def test_probability_writer_emits_no_object_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
    *,
    custom_model: bool,
) -> None:
    import cleanlab_extract_probs
    import torch
    from torch.utils.data import DataLoader

    from torchgeo_bench.presets import ModelPreset

    model_name = "custom-rcf" if custom_model else "rcf"
    results = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "dataset": dataset,
                "method": "linear",
                "name": model_name,
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
    argv = [
        "cleanlab_extract_probs.py",
        "--dataset",
        dataset,
        "--results",
        str(results),
        "--out",
        str(output),
        "--device",
        "cpu",
    ]
    if custom_model:
        config = tmp_path / "run.yaml"
        config.write_text(
            "model:\n"
            "  name: custom-rcf\n"
            "  target: torchgeo_bench.models.RCFBench\n"
            "  kwargs:\n"
            "    features: 8\n"
            "    seed: 17\n"
            "    mode: empirical\n"
            f"datasets: [{dataset}]\n"
            "input:\n"
            "  image_size: null\n"
            "  time_steps: 2\n"
        )
        argv.extend(["--config", str(config)])
    monkeypatch.setattr("sys.argv", argv)
    labels = np.array([0, 1, 0, 1], dtype=np.int64)
    if dataset == "m-bigearthnet":
        labels = np.eye(2, dtype=np.float32)[labels]
    samples = [
        {"image": torch.zeros(3, 2, 2), "label": torch.from_numpy(np.asarray(label))}
        for label in labels
    ]
    loader = DataLoader(samples, batch_size=2)

    def datasets(**kwargs: object) -> tuple:
        assert kwargs["dataset_name"] == dataset
        assert kwargs["image_size"] == 16
        assert kwargs["interpolation"] == "bilinear"
        assert kwargs["bands"] == "rgb"
        assert kwargs["partition_name"] == "default"
        assert kwargs["time_steps"] == (2 if custom_model else None)
        return samples, loader, loader, loader

    def build(preset: ModelPreset, **kwargs: object) -> torch.nn.Module:
        assert preset.name == model_name
        assert kwargs["normalization"] == "bandspec_zscore"
        assert preset.kwargs["seed"] == (17 if custom_model else 0)
        if custom_model:
            assert kwargs["dataset"] is samples
            assert preset.kwargs["features"] == 8
        else:
            assert "dataset" not in kwargs
        return torch.nn.Identity()

    monkeypatch.setattr(cleanlab_extract_probs, "get_datasets", datasets)
    monkeypatch.setattr("torchgeo_bench.main.build_model", build)
    monkeypatch.setattr(
        cleanlab_extract_probs,
        "embed_split",
        lambda *args, **kwargs: (np.zeros((4, 2), dtype=np.float32), labels),
    )

    class Probe:
        def __init__(self, **kwargs: object) -> None:
            self.classes_ = np.array([0, 1], dtype=np.int64)

        def fit(self, images: torch.Tensor, labels: torch.Tensor) -> None:
            pass

        def predict_proba(self, images: torch.Tensor) -> np.ndarray:
            return np.tile(np.array([[0.75, 0.25]], dtype=np.float32), (len(images), 1))

    monkeypatch.setattr(cleanlab_extract_probs, "LogisticRegression", Probe)
    cleanlab_extract_probs.main()
    for split in ("train", "test"):
        path = output / f"{dataset}__{model_name}_{split}.npz"
        with np.load(path, allow_pickle=False) as archive:
            assert all(not archive[name].dtype.hasobject for name in archive.files)
            assert archive["meta"].dtype.kind == "U"
            assert archive["meta"][1] == model_name
            assert archive["meta"][-1] == split
            np.testing.assert_array_equal(archive["indices"], np.arange(len(labels)))
            np.testing.assert_array_equal(archive["labels"], labels)
        np.testing.assert_array_equal(run_cleanlab_audit._load_npz(path)["labels"], labels)
