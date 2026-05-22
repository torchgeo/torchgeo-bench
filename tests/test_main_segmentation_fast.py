"""Fast offline tests for segmentation orchestration in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from torchgeo_bench.main import main

from .test_main_fast import _compose_cfg


class _SegmentationDataset(Dataset):
    """Small dataset wrapper that emits ``{"image", "mask"}`` samples."""

    def __init__(self, images: torch.Tensor, masks: torch.Tensor) -> None:
        self._images = images
        self._masks = masks

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[index], "mask": self._masks[index]}


def _synthetic_segmentation_loaders(
    n_train: int = 8,
    n_val: int = 4,
    n_test: int = 4,
    channels: int = 3,
    n_classes: int = 3,
) -> tuple[_SegmentationDataset, DataLoader, DataLoader, DataLoader]:
    """Return train dataset + train/val/test loaders for segmentation."""
    rng = torch.Generator().manual_seed(1)
    train_images = torch.rand(n_train, channels, 64, 64, generator=rng)
    val_images = torch.rand(n_val, channels, 64, 64, generator=rng)
    test_images = torch.rand(n_test, channels, 64, 64, generator=rng)

    train_masks = torch.randint(0, n_classes, (n_train, 64, 64), generator=rng)
    val_masks = torch.randint(0, n_classes, (n_val, 64, 64), generator=rng)
    test_masks = torch.randint(0, n_classes, (n_test, 64, 64), generator=rng)

    train_dataset = _SegmentationDataset(train_images, train_masks)
    val_dataset = _SegmentationDataset(val_images, val_masks)
    test_dataset = _SegmentationDataset(test_images, test_masks)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_loader, test_loader


def _seg_resume_row(cfg, *, metric_name: str = "mIoU") -> dict[str, object]:
    return {
        "dataset": "burn_scars",
        "method": "seg-fpn",
        "model": cfg.model._target_,
        "name": cfg.model.name,
        "normalization": cfg.dataset.normalization,
        "image_size": cfg.dataset.image_size,
        "interpolation": cfg.dataset.interpolation,
        "partition": cfg.dataset.partition,
        "bands": cfg.dataset.bands,
        "metric_name": metric_name,
        "metric_value": 0.42,
    }


def _cfg_for_segmentation(out: Path, overrides: list[str] | None = None):
    return _compose_cfg(
        out,
        overrides=[
            "dataset.names=[burn_scars]",
            "eval.segmentation.cache_features=false",
            "eval.segmentation.head_type=fpn",
            "eval.segmentation.save_viz=false",
            *(overrides or []),
        ],
    )


def _mock_probe_and_solver():
    probe = mock.Mock()
    probe.channels_list = [16, 32]
    solver = mock.Mock()
    solver.fit.return_value = None
    solver.evaluate.return_value = {
        "mIoU": 0.42,
        "fw_IoU": 0.55,
        "precision": 0.6,
        "recall": 0.7,
        "f1": 0.65,
    }
    return probe, solver


def test_segmentation_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(),
        ),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    assert df["method"].str.startswith("seg-").any()
    assert "miou" in set(df["metric_name"].str.lower())


def test_segmentation_viz_not_called_when_disabled(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["eval.segmentation.save_viz=false"])

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(),
        ),
        mock.patch("torchgeo_bench.main.save_segmentation_viz") as viz_mock,
    ):
        main.__wrapped__(cfg)

    viz_mock.assert_not_called()


def test_segmentation_resume_skips_complete_run(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["resume=true"])
    pd.DataFrame([_seg_resume_row(cfg)]).to_csv(out, index=False)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch("torchgeo_bench.main._build_seg_probe_and_solver") as build_mock,
    ):
        main.__wrapped__(cfg)

    build_mock.assert_not_called()
    df = pd.read_csv(out)
    assert int((df["method"] == "seg-fpn").sum()) == 1


def test_segmentation_viz_called_when_enabled(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(
        out,
        overrides=["eval.segmentation.save_viz=true", "eval.segmentation.n_viz_samples=2"],
    )
    probe, solver = _mock_probe_and_solver()
    preds = torch.zeros(4, 64, 64, dtype=torch.long)
    solver.evaluate.return_value = (
        {
            "mIoU": 0.42,
            "fw_IoU": 0.55,
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
        },
        preds,
    )

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch("torchgeo_bench.main._build_seg_probe_and_solver", return_value=(probe, solver)),
        mock.patch("torchgeo_bench.main.save_segmentation_viz") as viz_mock,
    ):
        main.__wrapped__(cfg)

    viz_mock.assert_called_once()
