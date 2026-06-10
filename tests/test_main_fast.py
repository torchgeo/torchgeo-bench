"""Fast offline tests for classification orchestration in ``torchgeo_bench.main``."""

from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import DatasetNotFoundError

from torchgeo_bench.main import main


class _DictTensorDataset(Dataset):
    """Small dataset wrapper that emits ``{"image", "label"}`` samples."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self._images[index],
            "label": self._labels[index],
        }


def _compose_cfg(output_path: Path, overrides: Sequence[str] | None = None) -> DictConfig:
    """Compose Hydra config for fast offline main-path tests."""
    extra = list(overrides or [])
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        return compose(
            config_name="config",
            overrides=[
                "model=rcf",
                "dataset.names=[m-eurosat]",
                "dataset.partition=default",
                "dataset.batch_size=4",
                "dataset.num_workers=0",
                "eval.bootstrap=5",
                "eval.c_range=[-2,-1,2]",
                "device=cpu",
                f"output={output_path}",
                *extra,
            ],
        )


def _synthetic_loaders(
    n_train: int = 16,
    n_val: int = 8,
    n_test: int = 8,
    n_classes: int = 10,
    channels: int = 3,
) -> tuple[_DictTensorDataset, DataLoader, DataLoader, DataLoader]:
    """Return train dataset + train/val/test loaders matching benchmark contract."""
    rng = torch.Generator().manual_seed(0)
    train_images = torch.rand(n_train, channels, 64, 64, generator=rng) * 3000.0
    val_images = torch.rand(n_val, channels, 64, 64, generator=rng) * 3000.0
    test_images = torch.rand(n_test, channels, 64, 64, generator=rng) * 3000.0

    train_labels = torch.randint(0, n_classes, (n_train,), generator=rng)
    val_labels = torch.randint(0, n_classes, (n_val,), generator=rng)
    test_labels = torch.randint(0, n_classes, (n_test,), generator=rng)

    train_dataset = _DictTensorDataset(train_images, train_labels)
    val_dataset = _DictTensorDataset(val_images, val_labels)
    test_dataset = _DictTensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_loader, test_loader


def _synthetic_embeddings() -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic (X, y) tuples for train/val/test ``embed_split`` calls."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((16, 8), dtype=np.float32)
    y_train = rng.integers(0, 10, size=(16,), dtype=np.int64)
    x_val = rng.standard_normal((8, 8), dtype=np.float32)
    y_val = rng.integers(0, 10, size=(8,), dtype=np.int64)
    x_test = rng.standard_normal((8, 8), dtype=np.float32)
    y_test = rng.integers(0, 10, size=(8,), dtype=np.int64)
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def _resume_row(cfg: DictConfig, *, method: str, metric_name: str) -> dict[str, object]:
    """Build a resume-key-matching CSV row for pre-seeding output files."""
    return {
        "dataset": "m-eurosat",
        "method": method,
        "model": cfg.model._target_,
        "name": cfg.model.name,
        "normalization": cfg.dataset.normalization,
        "image_size": cfg.dataset.image_size,
        "interpolation": cfg.dataset.interpolation,
        "partition": cfg.dataset.partition,
        "bands": cfg.dataset.bands,
        "metric_name": metric_name,
        "metric_value": 0.1,
    }


def test_knn_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["eval.skip_linear=true"])

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    assert "knn5" in df["method"].values
    row = df[df["method"] == "knn5"].iloc[0]
    assert row["metric_name"] == "accuracy"
    assert row["dataset"] == "m-eurosat"


def test_linear_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
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
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    assert "linear" in df["method"].values
    row = df[df["method"] == "linear"].iloc[0]
    assert row["metric_name"] == "accuracy"


def test_resume_skips_completed_knn_row(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["resume=true", "eval.skip_linear=true"])
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
    ):
        main.__wrapped__(cfg)

    knn_mock.assert_not_called()
    df = pd.read_csv(out)
    assert int((df["method"] == "knn5").sum()) == 1


def test_dataset_not_found_skips(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)

    with mock.patch(
        "torchgeo_bench.main.get_datasets", side_effect=DatasetNotFoundError("missing")
    ):
        main.__wrapped__(cfg)

    assert not out.exists()


def test_csv_row_has_required_columns(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["eval.skip_linear=true"])

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    required = {"dataset", "method", "model", "metric_name", "metric_value", "partition", "bands"}
    assert required.issubset(set(df.columns))
