"""Offline tests for the multilabel runner."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.main import LinearProbeDivergedError, main
from torchgeo_bench.presets import merge_settings

from .test_main_fast import _compose_cfg, _DictTensorDataset, _resume_row


def _synthetic_multilabel_loaders(
    n_train: int = 12,
    n_val: int = 6,
    n_test: int = 6,
    channels: int = 3,
    n_classes: int = 8,
) -> tuple[_DictTensorDataset, DataLoader, DataLoader, DataLoader]:
    rng = torch.Generator().manual_seed(2)
    train_images = torch.rand(n_train, channels, 64, 64, generator=rng) * 3000.0
    val_images = torch.rand(n_val, channels, 64, 64, generator=rng) * 3000.0
    test_images = torch.rand(n_test, channels, 64, 64, generator=rng) * 3000.0

    train_labels = torch.randint(
        0, 2, (n_train, n_classes), generator=rng, dtype=torch.int64
    ).float()
    val_labels = torch.randint(0, 2, (n_val, n_classes), generator=rng, dtype=torch.int64).float()
    test_labels = torch.randint(0, 2, (n_test, n_classes), generator=rng, dtype=torch.int64).float()

    train_dataset = _DictTensorDataset(train_images, train_labels)
    val_dataset = _DictTensorDataset(val_images, val_labels)
    test_dataset = _DictTensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_loader, test_loader


def _synthetic_multilabel_embeddings() -> list[tuple[np.ndarray, np.ndarray]]:
    """Embeddings in the order of the train, validation, and test calls."""
    rng = np.random.default_rng(2)
    x_train = rng.standard_normal((12, 10), dtype=np.float32)
    y_train = rng.integers(0, 2, size=(12, 8)).astype(np.float32)
    x_val = rng.standard_normal((6, 10), dtype=np.float32)
    y_val = rng.integers(0, 2, size=(6, 8)).astype(np.float32)
    x_test = rng.standard_normal((6, 10), dtype=np.float32)
    y_test = rng.integers(0, 2, size=(6, 8)).astype(np.float32)
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def _multilabel_resume_row(cfg) -> dict[str, object]:
    return {
        **_resume_row(cfg, method="knn5", metric_name="micro_mAP"),
        "dataset": "m-bigearthnet",
        "num_classes": 43,
        "metric_value": 0.2,
    }


def _cfg_for_multilabel(out: Path, overrides: dict | None = None):
    return _compose_cfg(
        out,
        overrides=merge_settings({"datasets": ["m-bigearthnet"]}, overrides or {}),
    )


def test_multilabel_knn_emits_micro_map(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_multilabel(out, overrides={"classification": {"methods": ["knn"]}})

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_multilabel_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main.embed_split", side_effect=_synthetic_multilabel_embeddings()
        ),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    row = df[df["method"] == "knn5"].iloc[0]
    assert row["metric_name"] == "micro_mAP"


def test_multilabel_linear_emits_micro_map(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_multilabel(out)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_multilabel_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main.embed_split", side_effect=_synthetic_multilabel_embeddings()
        ),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch(
            "torchgeo_bench.main.evaluate_logistic",
            return_value=(
                0.6,
                0.52,
                0.66,
                0.1,
                {"ece": 0.04, "rms_ce": 0.06, "mce": 0.09},
                {"ece_ts": 0.04, "rms_ce_ts": 0.06, "mce_ts": 0.09, "temperature": 0.8},
            ),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    row = df[df["method"] == "linear"].iloc[0]
    assert row["metric_name"] == "micro_mAP"


def test_diverged_linear_probe_skips_row_not_whole_run(tmp_path: Path):
    """A diverged linear probe must not discard the KNN result for the same dataset."""
    out = tmp_path / "out.csv"
    cfg = _cfg_for_multilabel(out)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_multilabel_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main.embed_split", side_effect=_synthetic_multilabel_embeddings()
        ),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch(
            "torchgeo_bench.main.evaluate_logistic",
            side_effect=LinearProbeDivergedError("every candidate C diverged"),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    assert (df["method"] == "knn5").any()
    assert not (df["method"] == "linear").any()


def test_multilabel_resume_key_stable(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_multilabel(
        out, overrides={"output": {"resume": True}, "classification": {"methods": ["knn"]}}
    )
    pd.DataFrame([_multilabel_resume_row(cfg)]).to_csv(out, index=False)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_multilabel_loaders()
        ),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    df = pd.read_csv(out)
    assert int((df["method"] == "knn5").sum()) == 1
