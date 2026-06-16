"""Fast offline tests for segmentation orchestration in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
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
        "feature_norm": cfg.eval.feature_norm,
        "solver": cfg.eval.solver,
        "metric_name": metric_name,
        "metric_value": 0.42,
    }


def _image_stats_paths(tmp_path: Path) -> tuple[Path, Path]:
    return (
        tmp_path / "results" / "all_segmentation_results.csv",
        tmp_path / "results" / "segmentation_image_stats.csv",
    )


def _seg_image_stats_rows(
    cfg, *, n_images: int = 4, offset: float = 0.0
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for image_index in range(n_images):
        rows.append(
            {
                "model": cfg.model._target_,
                "name": cfg.model.name,
                "dataset": "burn_scars",
                "partition": cfg.dataset.partition,
                "seed": cfg.seed,
                "normalization": cfg.dataset.normalization,
                "bands": cfg.dataset.bands,
                "image_size": cfg.dataset.image_size,
                "interpolation": cfg.dataset.interpolation,
                "seg_head_type": cfg.eval.segmentation.head_type,
                "seg_layers": "",
                "seg_epochs": cfg.eval.segmentation.epochs,
                "seg_lr": cfg.eval.segmentation.lr,
                "seg_batch_size": cfg.eval.segmentation.batch_size,
                "seg_cache_features": cfg.eval.segmentation.cache_features,
                "seg_cache_dtype": cfg.eval.segmentation.cache_dtype,
                "seg_ignore_index": cfg.eval.segmentation.criterion.ignore_index,
                "seg_lr_scheduler": cfg.eval.segmentation.lr_scheduler,
                "image_index": image_index,
                "height": 64,
                "width": 64,
                "valid_pixel_count": 4096,
                "ignored_pixel_count": 0,
                "n_gt_classes": 2,
                "n_pred_classes": 2,
                "n_pred_or_gt_classes": 2,
                "image_pixel_accuracy": 0.5 + offset,
                "image_miou_gt_present": 0.25 + offset,
                "image_miou_pred_or_gt_present": 0.25 + offset,
                "mean_1mp": 0.1 + offset,
                "median_1mp": 0.1 + offset,
                "mean_entropy": 0.2 + offset,
                "median_entropy": 0.2 + offset,
                "mean_normalized_entropy": 0.3 + offset,
                "median_normalized_entropy": 0.3 + offset,
                "pixel_error_aupr_1mp": 0.4 + offset,
                "pixel_error_auroc_1mp": 0.5 + offset,
                "pixel_error_aupr_entropy": 0.6 + offset,
                "pixel_error_auroc_entropy": 0.7 + offset,
            }
        )
    return rows


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


def _mock_probe_and_solver(cfg, *, n_images: int = 4):
    probe = mock.Mock()
    probe.channels_list = [16, 32]
    solver = mock.Mock()
    solver.fit.return_value = None
    metrics = {
        "mIoU": 0.42,
        "fw_IoU": 0.55,
        "precision": 0.6,
        "recall": 0.7,
        "f1": 0.65,
    }
    preds = torch.zeros(n_images, 64, 64, dtype=torch.long)
    image_stats_rows = _seg_image_stats_rows(cfg, n_images=n_images)

    def _evaluate(*args, collect_preds=False, collect_image_stats=False, **kwargs):
        del args, kwargs
        rows = [row.copy() for row in image_stats_rows]
        if collect_preds and collect_image_stats:
            return metrics, preds, rows
        if collect_preds:
            return metrics, preds
        if collect_image_stats:
            return metrics, rows
        return metrics

    solver.evaluate.side_effect = _evaluate
    solver.evaluate_cached.side_effect = _evaluate
    return probe, solver


def test_segmentation_row_emitted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out)
    summary_csv, image_stats_csv = _image_stats_paths(tmp_path)
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(cfg),
        ),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(summary_csv)
    assert df["method"].str.startswith("seg-").any()
    assert "miou" in set(df["metric_name"].str.lower())
    image_stats_df = pd.read_csv(image_stats_csv)
    assert len(image_stats_df) == 4
    assert "image_index" in image_stats_df.columns
    assert not out.exists()


def test_segmentation_viz_not_called_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["eval.segmentation.save_viz=false"])
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(cfg),
        ),
        mock.patch("torchgeo_bench.main.save_segmentation_viz") as viz_mock,
    ):
        main.__wrapped__(cfg)

    viz_mock.assert_not_called()


def test_segmentation_resume_skips_complete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["resume=true"])
    summary_csv, image_stats_csv = _image_stats_paths(tmp_path)
    monkeypatch.chdir(tmp_path)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_seg_resume_row(cfg)]).to_csv(summary_csv, index=False)
    pd.DataFrame(_seg_image_stats_rows(cfg)).to_csv(image_stats_csv, index=False)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch("torchgeo_bench.main._build_seg_probe_and_solver") as build_mock,
    ):
        main.__wrapped__(cfg)

    build_mock.assert_not_called()
    df = pd.read_csv(summary_csv)
    assert int((df["method"] == "seg-fpn").sum()) == 1


def test_segmentation_resume_reruns_when_image_stats_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["resume=true"])
    summary_csv, image_stats_csv = _image_stats_paths(tmp_path)
    monkeypatch.chdir(tmp_path)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_seg_resume_row(cfg)]).to_csv(summary_csv, index=False)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(cfg),
        ) as build_mock,
    ):
        main.__wrapped__(cfg)

    build_mock.assert_called_once()
    summary_df = pd.read_csv(summary_csv)
    assert len(summary_df) == 1
    assert len(pd.read_csv(image_stats_csv)) == 4


def test_segmentation_resume_replaces_incomplete_image_stats_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["resume=true"])
    summary_csv, image_stats_csv = _image_stats_paths(tmp_path)
    monkeypatch.chdir(tmp_path)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_seg_resume_row(cfg)]).to_csv(summary_csv, index=False)
    pd.DataFrame(_seg_image_stats_rows(cfg, n_images=2, offset=0.9)).to_csv(
        image_stats_csv, index=False
    )

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(cfg),
        ),
    ):
        main.__wrapped__(cfg)

    image_stats_df = pd.read_csv(image_stats_csv)
    assert len(image_stats_df) == 4
    assert set(image_stats_df["image_index"]) == {0, 1, 2, 3}
    assert float(image_stats_df["image_pixel_accuracy"].iloc[0]) == pytest.approx(0.5)


def test_segmentation_image_stats_overwrite_replaces_existing_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["eval.segmentation.image_stats.overwrite=true"])
    _, image_stats_csv = _image_stats_paths(tmp_path)
    monkeypatch.chdir(tmp_path)
    image_stats_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_seg_image_stats_rows(cfg, offset=0.9)).to_csv(image_stats_csv, index=False)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.main._build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(cfg),
        ),
    ):
        main.__wrapped__(cfg)

    image_stats_df = pd.read_csv(image_stats_csv)
    assert len(image_stats_df) == 4
    assert float(image_stats_df["image_pixel_accuracy"].iloc[0]) == pytest.approx(0.5)


def test_segmentation_viz_called_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(
        out,
        overrides=["eval.segmentation.save_viz=true", "eval.segmentation.n_viz_samples=2"],
    )
    probe, solver = _mock_probe_and_solver(cfg)
    monkeypatch.chdir(tmp_path)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch("torchgeo_bench.main._build_seg_probe_and_solver", return_value=(probe, solver)),
        mock.patch("torchgeo_bench.main.save_segmentation_viz") as viz_mock,
    ):
        main.__wrapped__(cfg)

    viz_mock.assert_called_once()
