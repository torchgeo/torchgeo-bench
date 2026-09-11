"""Small deterministic inputs for runner orchestration tests."""

from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from torchgeo_bench.config import compose_config
from torchgeo_bench.resume import _resume_config_hash


class _DictTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[index], "label": self._labels[index]}


def _compose_cfg(output_path: Path, overrides: Sequence[str] = ()) -> DictConfig:
    return compose_config(
        [
            "model=rcf",
            "dataset.names=[m-eurosat]",
            "dataset.partition=default",
            "dataset.batch_size=4",
            "dataset.num_workers=0",
            "eval.bootstrap=5",
            "eval.c_range=[-2,-1,2]",
            "device=cpu",
            f"output={output_path}",
            *overrides,
        ]
    )


def _synthetic_loaders() -> tuple[_DictTensorDataset, DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator().manual_seed(0)
    datasets = [
        _DictTensorDataset(
            torch.rand(size, 3, 8, 8, generator=generator) * 3000,
            torch.arange(size) % 10,
        )
        for size in (16, 8, 8)
    ]
    loaders = [DataLoader(dataset, batch_size=4, num_workers=0) for dataset in datasets]
    return datasets[0], *loaders


def _synthetic_embeddings() -> list[tuple[np.ndarray, np.ndarray]]:
    generator = np.random.default_rng(0)
    return [
        (
            generator.standard_normal((size, 8), dtype=np.float32),
            np.arange(size, dtype=np.int64) % 10,
        )
        for size in (16, 8, 8)
    ]


def _resume_row(cfg: DictConfig, *, method: str, metric_name: str) -> dict[str, object]:
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
        "num_classes": 10,
        "config_hash": _resume_config_hash(cfg),
        "metric_name": metric_name,
        "metric_value": 0.1,
    }


def _chainable_model_mock() -> mock.Mock:
    model = mock.Mock()
    model.to.return_value = model
    model.eval.return_value = model
    return model
