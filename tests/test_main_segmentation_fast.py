"""Fast offline tests for segmentation orchestration in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from torchgeo_bench.main import main
from torchgeo_bench.resume import _resume_config_hash

from .test_main_fast import _chainable_model_mock, _compose_cfg


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
        "num_classes": 3,
        "config_hash": _resume_config_hash(cfg),
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
    metrics = {
        "mIoU": 0.42,
        "fw_IoU": 0.55,
        "precision": 0.6,
        "recall": 0.7,
        "f1": 0.65,
    }
    confusions = torch.tensor([[[0, 4], [0, 0]], [[0, 0], [0, 4]]])

    def evaluate(*_args, collect_preds: bool = False, collect_confusions: bool = False, **_kwargs):
        if collect_preds and collect_confusions:
            return metrics, torch.zeros(2, 64, 64, dtype=torch.long), confusions
        if collect_preds:
            return metrics, torch.zeros(2, 64, 64, dtype=torch.long)
        if collect_confusions:
            return metrics, confusions
        return metrics

    solver.evaluate.side_effect = evaluate
    return probe, solver


def test_segmentation_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out)

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    assert df["method"].str.startswith("seg-").any()
    assert "miou" in set(df["metric_name"].str.lower())
    assert df.loc[0, "best_lr"] == 1e-3
    assert df.loc[0, "best_batch_size"] == 2
    assert not df.loc[0, "merge_val"]
    assert df.loc[0, "ci_lower"] < df.loc[0, "ci_upper"]


def test_cached_segmentation_records_probe_batch_size(tmp_path: Path):
    """Cached probes use their own configured batch size, not loader batch size."""
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(
        out,
        overrides=["eval.segmentation.cache_features=true", "eval.segmentation.batch_size=3"],
    )
    probe, solver = _mock_probe_and_solver()
    cache = mock.Mock()
    probe.freeze_backbone = True
    probe.extract_segmentation_features.return_value = cache
    solver.evaluate_cached.return_value = (
        {
            "mIoU": 0.42,
            "fw_IoU": 0.55,
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
        },
        torch.tensor([[[0, 4], [0, 0]], [[0, 0], [0, 4]]]),
    )

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
            return_value=(probe, solver),
        ),
    ):
        main(cfg)

    solver.fit_cached.assert_called_once_with(
        train_cache=cache,
        val_cache=cache,
        batch_size=3,
        epochs=cfg.eval.segmentation.epochs,
        verbose=cfg.verbose,
    )
    df = pd.read_csv(out)
    assert df.loc[0, "best_batch_size"] == 3


def test_segmentation_viz_not_called_when_disabled(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["eval.segmentation.save_viz=false"])

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(),
        ),
        mock.patch("torchgeo_bench.segmentation_viz.save_segmentation_viz") as viz_mock,
    ):
        main(cfg)

    viz_mock.assert_not_called()


def test_failed_requested_viz_is_not_marked_complete(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["eval.segmentation.save_viz=true"])
    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
            return_value=_mock_probe_and_solver(),
        ),
        mock.patch(
            "torchgeo_bench.segmentation_viz.save_segmentation_viz",
            side_effect=RuntimeError("plot rendering failed"),
        ),
        pytest.raises(RuntimeError, match="plot rendering failed"),
    ):
        main(cfg)
    assert not out.exists()


def test_segmentation_resume_skips_complete_run(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _cfg_for_segmentation(out, overrides=["resume=true"])
    pd.DataFrame([_seg_resume_row(cfg)]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch("torchgeo_bench.main.get_datasets") as data_mock,
        mock.patch("torchgeo_bench.main.instantiate", return_value=model) as instantiate_mock,
        mock.patch("torchgeo_bench.segmentation_task.build_seg_probe_and_solver") as build_mock,
    ):
        main(cfg)

    data_mock.assert_not_called()
    instantiate_mock.assert_not_called()
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
    solver.evaluate.side_effect = None
    solver.evaluate.return_value = (
        {
            "mIoU": 0.42,
            "fw_IoU": 0.55,
            "precision": 0.6,
            "recall": 0.7,
            "f1": 0.65,
        },
        preds,
        torch.tensor([[[0, 4], [0, 0]], [[0, 0], [0, 4]]]),
    )

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_segmentation_loaders()
        ),
        mock.patch(
            "torchgeo_bench.segmentation_task.build_seg_probe_and_solver",
            return_value=(probe, solver),
        ),
        mock.patch("torchgeo_bench.segmentation_viz.save_segmentation_viz") as viz_mock,
    ):
        main(cfg)

    viz_mock.assert_called_once()
