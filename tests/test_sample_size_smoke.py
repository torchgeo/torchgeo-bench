"""Smoke tests for the sample-size sweep pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch.nn as nn
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf


class _MockSegBackbone(nn.Module):
    """Minimal two-layer CNN backbone for segmentation smoke tests."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)


class _DummyLoader:
    """Minimal loader stub that exposes a dataset length."""

    def __init__(self, length: int) -> None:
        self.dataset = list(range(length))


# ---------------------------------------------------------------------------
# Slice 2: Config composition
# ---------------------------------------------------------------------------


def test_sample_size_config_composes():
    """Hydra compose produces the expected default keys and values."""
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(
            config_name="sample_size_config",
            overrides=["model=rcf"],
        )
    assert list(cfg.sample_size.fractions) == [0.01, 0.10, 0.25, 0.50, 0.75]
    assert cfg.sample_size.seeds_cls == 5
    assert cfg.sample_size.seeds_seg == 3
    assert cfg.sample_size.target_grad_steps == 1000
    assert cfg.sample_size.image_stats.enabled is True
    assert "output" in cfg
    assert "resume" in cfg


# ---------------------------------------------------------------------------
# Helpers shared by Slices 3–5
# ---------------------------------------------------------------------------


class _FakeCls:
    """Minimal BenchDataset stub for classification."""

    task = "classification"
    multilabel = False
    rgb_bands = [0, 1, 2]

    def select_band_specs(self, bands):
        return []


class _FakeSegCls:
    """Minimal BenchDataset stub for segmentation."""

    task = "segmentation"
    multilabel = False
    num_classes = 3
    rgb_bands = [0, 1, 2]

    def select_band_specs(self, bands):
        return []


@pytest.fixture
def _cls_embeddings():
    rng = np.random.default_rng(0)
    X_train = rng.standard_normal((100, 8)).astype(np.float32)
    y_train = np.repeat([0, 1], 50).astype(np.int64)
    X_val = rng.standard_normal((20, 8)).astype(np.float32)
    y_val = np.repeat([0, 1], 10).astype(np.int64)
    X_test = rng.standard_normal((20, 8)).astype(np.float32)
    y_test = np.repeat([0, 1], 10).astype(np.int64)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def _make_cls_cfg(
    tmp_path, *, fractions=(0.05, 0.50), seeds_cls=2, resume=False, output_name="ss.csv"
):
    return OmegaConf.create(
        {
            "model": {"_target_": "dummy.T", "name": "resnet50"},
            "dataset": {
                "names": ["m-eurosat"],
                "partition": "default",
                "batch_size": 2,
                "num_workers": 0,
                "bands": "rgb",
                "normalization": "bandspec_zscore",
                "interpolation": "bilinear",
            },
            "sample_size": {
                "fractions": list(fractions),
                "seeds_cls": seeds_cls,
                "seeds_seg": 3,
                "target_grad_steps": 2000,
                "c_range": [-2, -1, 0, 1, 2],
                "n_bins_ece": 15,
                "merge_val": False,
                "image_stats": {
                    "enabled": True,
                    "root": str(tmp_path / "cls_image_stats"),
                    "format": "parquet",
                    "overwrite": False,
                },
            },
            "output": str(tmp_path / output_name),
            "resume": resume,
            "device": "cpu",
            "verbose": False,
            "seed": 0,
        }
    )


def _patch_cls(monkeypatch, _cls_embeddings):
    train_split, val_split, test_split = _cls_embeddings
    embed_iter = iter([train_split, val_split])
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.embed_split",
        lambda *a, **kw: next(embed_iter),
    )
    sample_ids = np.array([f"test-{idx}" for idx in range(len(test_split[1]))], dtype=object)
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline._embed_test_split_with_ids",
        lambda *a, **kw: (test_split[0], test_split[1], sample_ids),
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.get_bench_dataset_class",
        lambda _: _FakeCls,
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.get_datasets",
        lambda **_: (
            None,
            _DummyLoader(len(train_split[1])),
            _DummyLoader(len(val_split[1])),
            _DummyLoader(len(test_split[1])),
        ),
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.instantiate",
        lambda *a, **kw: nn.Identity(),
    )


# ---------------------------------------------------------------------------
# Slice 3: Classification path
# ---------------------------------------------------------------------------


def test_classification_path_writes_csv(tmp_path, monkeypatch, _cls_embeddings):
    """Classification sweep writes CSV with correct row count and valid values."""
    _patch_cls(monkeypatch, _cls_embeddings)

    from torchgeo_bench.sample_size_pipeline import main as ss_main

    cfg = _make_cls_cfg(tmp_path)
    ss_main.__wrapped__(cfg)

    csv_path = tmp_path / "ss.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    expected_metrics = {
        "accuracy",
        "ece",
        "signed_ece",
        "nll",
        "brier",
        "mean_confidence",
        "overconfidence_gap",
        "mean_wrong_confidence",
        "high_conf_wrong_rate_090",
        "selective_acc_90",
        "raw_aurc",
        "eaurc",
    }
    # 2 fractions × 2 seeds × 12 metrics = 48 rows
    assert len(df) == 48
    assert set(df["metric_name"]) == expected_metrics
    assert df["metric_value"].notna().all()
    sub = df.loc[df["train_fraction"] == 0.05, "n_train_used"]
    full = df.loc[df["train_fraction"] == 0.05, "n_train_full"]
    assert sub.lt(full).all()

    wide = df.pivot_table(
        index=["train_fraction", "seed"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    )
    assert wide["mean_confidence"].between(0.0, 1.0).all()
    assert wide["high_conf_wrong_rate_090"].between(0.0, 1.0).all()
    assert wide["selective_acc_90"].between(0.0, 1.0).all()
    assert wide["signed_ece"].between(0.0, 1.0).all()
    assert wide["raw_aurc"].ge(0.0).all()
    assert wide["eaurc"].ge(0.0).all()
    assert wide["mean_wrong_confidence"].dropna().between(0.0, 1.0).all()
    image_stats_paths = sorted((tmp_path / "cls_image_stats").rglob("*.parquet"))
    assert len(image_stats_paths) == 4
    image_stats_df = pd.read_parquet(image_stats_paths[0])
    assert len(image_stats_df) == 20
    assert {"image_index", "sample_id", "confidence", "margin", "entropy", "nll"}.issubset(
        image_stats_df.columns
    )
    assert image_stats_df["sample_id"].iloc[0] == "test-0"


def test_classification_csv_schema(tmp_path, monkeypatch, _cls_embeddings):
    """Classification sweep CSV has the required column schema."""
    _patch_cls(monkeypatch, _cls_embeddings)

    from torchgeo_bench.sample_size_pipeline import main as ss_main

    cfg = _make_cls_cfg(tmp_path, output_name="ss_schema.csv")
    ss_main.__wrapped__(cfg)

    df = pd.read_csv(tmp_path / "ss_schema.csv")
    required = {
        "model",
        "dataset",
        "train_fraction",
        "seed",
        "task",
        "metric_name",
        "metric_value",
        "n_train_full",
        "n_train_used",
        "n_val",
        "n_test",
        "best_c",
    }
    assert required.issubset(set(df.columns))


def test_compute_cls_metrics_handles_zero_errors():
    from torchgeo_bench.sample_size_pipeline import _compute_cls_metrics

    y_true = np.array([0, 1], dtype=np.int64)
    probs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    metrics = _compute_cls_metrics(y_true, probs, n_bins_ece=15)

    assert metrics["accuracy"] == 1.0
    assert metrics["mean_confidence"] == 1.0
    assert metrics["overconfidence_gap"] == 0.0
    assert metrics["high_conf_wrong_rate_090"] == 0.0
    assert metrics["selective_acc_90"] == 1.0
    assert metrics["raw_aurc"] == 0.0
    assert metrics["eaurc"] == 0.0
    assert np.isnan(metrics["mean_wrong_confidence"])


# ---------------------------------------------------------------------------
# Slice 4: Segmentation path
# ---------------------------------------------------------------------------


def _make_fake_cache(n=20, n_layers=1, c=16, h=8, w=8):
    """Build a synthetic CachedFeaturesDataset."""
    import torch

    from torchgeo_bench.segmentation_probe import CachedFeaturesDataset

    layer_tensors = [torch.randn(n, c, h, w) for _ in range(n_layers)]
    masks = torch.randint(0, 3, (n, h, w))
    return CachedFeaturesDataset(layer_tensors, masks)


def _make_seg_cfg(
    tmp_path, *, fractions=(0.50, 1.0), seeds_seg=2, resume=False, output_name="ss_seg.csv"
):
    return OmegaConf.create(
        {
            "model": {"_target_": "dummy.T", "name": "resnet50"},
            "dataset": {
                "names": ["burn_scars"],
                "partition": "default",
                "batch_size": 2,
                "num_workers": 0,
                "bands": "rgb",
                "normalization": "bandspec_zscore",
                "interpolation": "bilinear",
            },
            "sample_size": {
                "fractions": list(fractions),
                "seeds_cls": 5,
                "seeds_seg": seeds_seg,
                "target_grad_steps": 2000,
                "c_range": [-2, -1, 0, 1, 2],
                "n_bins_ece": 15,
                "merge_val": False,
                "image_stats": {
                    "enabled": True,
                    "root": str(tmp_path / "seg_image_stats"),
                    "format": "parquet",
                    "overwrite": False,
                },
            },
            "eval": {
                "segmentation": {
                    "layers": ["layer1"],
                    "head_type": "linear",
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "batch_size": 4,
                    "lr_scheduler": "none",
                    "criterion": {
                        "_target_": "torch.nn.CrossEntropyLoss",
                        "ignore_index": 255,
                    },
                }
            },
            "output": str(tmp_path / output_name),
            "resume": resume,
            "device": "cpu",
            "verbose": False,
            "seed": 0,
        }
    )


def test_segmentation_path_writes_csv(tmp_path, monkeypatch):
    """Segmentation sweep writes CSV with correct row count and valid values."""
    train_cache = _make_fake_cache(20)
    val_cache = _make_fake_cache(10)
    test_cache = _make_fake_cache(10)
    image_rows = [
        {
            "image_index": idx,
            "height": 8,
            "width": 8,
            "valid_pixel_count": 64,
            "ignored_pixel_count": 0,
            "n_gt_classes": 2,
            "n_pred_classes": 2,
            "n_pred_or_gt_classes": 2,
            "image_pixel_accuracy": 0.4,
            "image_miou_gt_present": 0.3,
            "image_miou_pred_or_gt_present": 0.3,
            "mean_1mp": 0.2,
            "median_1mp": 0.2,
            "mean_entropy": 0.1,
            "median_entropy": 0.1,
            "mean_normalized_entropy": 0.1,
            "median_normalized_entropy": 0.1,
            "pixel_error_aupr_1mp": 0.5,
            "pixel_error_auroc_1mp": 0.5,
            "pixel_error_aupr_entropy": 0.5,
            "pixel_error_auroc_entropy": 0.5,
        }
        for idx in range(10)
    ]

    cache_iter = iter([train_cache, val_cache, test_cache])
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.SegmentationProbe.extract_segmentation_features",
        lambda self, loader, **kw: next(cache_iter),
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.SegmentationSolver.fit_cached",
        lambda self, *a, **kw: 0.4,
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.SegmentationSolver.evaluate_cached",
        lambda self, *a, collect_image_stats=False, **kw: (
            (
                {
                    "mIoU": 0.4,
                    "ece": 0.1,
                    "pixel_ece": 0.1,
                    "fw_IoU": 0.4,
                    "precision": 0.4,
                    "recall": 0.4,
                    "f1": 0.4,
                },
                [row.copy() for row in image_rows],
            )
            if collect_image_stats
            else {
                "mIoU": 0.4,
                "ece": 0.1,
                "pixel_ece": 0.1,
                "fw_IoU": 0.4,
                "precision": 0.4,
                "recall": 0.4,
                "f1": 0.4,
            }
        ),
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.get_bench_dataset_class",
        lambda _: _FakeSegCls,
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.get_datasets",
        lambda **_: (None, None, None, None),
    )
    # Use a backbone with layer1/layer2 so SegmentationProbe.__init__ succeeds
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.instantiate",
        lambda *a, **kw: _MockSegBackbone(),
    )

    from torchgeo_bench.sample_size_pipeline import main as ss_main

    cfg = _make_seg_cfg(tmp_path)
    ss_main.__wrapped__(cfg)

    csv_path = tmp_path / "ss_seg.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    # 2 fractions × 2 seeds × 2 metrics (miou, pixel_ece) = 8 rows
    assert len(df) == 8
    assert df["metric_value"].notna().all()
    assert set(df["metric_name"]) == {"miou", "pixel_ece"}
    image_stats_paths = sorted((tmp_path / "seg_image_stats").rglob("*.parquet"))
    assert len(image_stats_paths) == 4
    image_stats_df = pd.read_parquet(image_stats_paths[0])
    assert len(image_stats_df) == 10
    assert "image_pixel_accuracy" in image_stats_df.columns


def test_epoch_scaling_formula():
    """_compute_epochs produces correct epoch counts from gradient-step target."""
    from torchgeo_bench.sample_size_pipeline import _compute_epochs

    # n_sub >= batch_size: steps_per_epoch = n_sub // batch_size
    assert _compute_epochs(n_sub=64, batch_size=64, target=100, floor=5) == 100
    assert _compute_epochs(n_sub=640, batch_size=64, target=100, floor=5) == 10
    # floor kicks in when formula < floor
    assert _compute_epochs(n_sub=6400, batch_size=64, target=100, floor=5) == 5
    # n_sub < batch_size: steps_per_epoch = 1
    assert _compute_epochs(n_sub=10, batch_size=64, target=10, floor=5) == 10


# ---------------------------------------------------------------------------
# Slice 5: Resume logic
# ---------------------------------------------------------------------------


def test_resume_skips_completed_cls_rows(tmp_path, monkeypatch, _cls_embeddings):
    """With resume=True and metrics plus image stats present, no extraction runs."""
    all_metrics = [
        "accuracy",
        "ece",
        "signed_ece",
        "nll",
        "brier",
        "mean_confidence",
        "overconfidence_gap",
        "mean_wrong_confidence",
        "high_conf_wrong_rate_090",
        "selective_acc_90",
        "raw_aurc",
        "eaurc",
    ]
    # Pre-seed CSV with all rows for 2 fractions × 2 seeds × 12 metrics.
    rows = [
        {
            "model": "resnet50",
            "dataset": "m-eurosat",
            "train_fraction": frac,
            "seed": seed,
            "task": "classification",
            "metric_name": metric,
            "metric_value": 0.5,
            "n_train_full": 100,
            "n_train_used": 5,
            "n_val": 20,
            "n_test": 20,
            "best_c": 1.0,
        }
        for frac in [0.05, 0.50]
        for seed in [0, 1]
        for metric in all_metrics
    ]
    csv_path = tmp_path / "resume.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    cfg = _make_cls_cfg(tmp_path, resume=True, output_name="resume.csv")

    call_count = []

    def _counting_embed(*a, **kw):
        call_count.append(1)
        return next(iter(_cls_embeddings))

    from torchgeo_bench.sample_size_pipeline import (
        _cls_image_stats_block_meta,
        _image_stats_block_key,
        _image_stats_block_path,
    )

    image_stats_root = Path(cfg.sample_size.image_stats.root)
    sample_ids = np.array([f"test-{idx}" for idx in range(20)], dtype=object)
    for fraction in [0.05, 0.50]:
        for seed in [0, 1]:
            meta = _cls_image_stats_block_meta(
                model_name="resnet50",
                model_target="dummy.T",
                dataset_name="m-eurosat",
                partition="default",
                bands="rgb",
                normalization="bandspec_zscore",
                image_size=None,
                interpolation="bilinear",
                train_fraction=fraction,
                seed=seed,
                n_train_full=100,
                n_train_used=100 if fraction >= 1.0 else max(1, int(np.floor(100 * fraction))),
                n_val=20,
                n_test=20,
            )
            block_key = _image_stats_block_key(meta)
            path = _image_stats_block_path(
                root=str(image_stats_root),
                task="classification",
                model="resnet50",
                dataset="m-eurosat",
                train_fraction=fraction,
                seed=seed,
                block_key=block_key,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        **meta,
                        "image_stats_block_key": block_key,
                        "best_c": 1.0,
                        "image_index": idx,
                        "sample_id": sample_ids[idx],
                        "y_true": int(idx % 2),
                        "y_pred": int(idx % 2),
                        "correct": True,
                        "confidence": 0.9,
                        "margin": 0.8,
                        "entropy": 0.1,
                        "normalized_entropy": 0.1,
                        "nll": 0.1,
                    }
                    for idx in range(20)
                ]
            ).to_parquet(path, index=False)

    monkeypatch.setattr("torchgeo_bench.sample_size_pipeline.embed_split", _counting_embed)
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline._embed_test_split_with_ids",
        lambda *a, **kw: (_cls_embeddings[2][0], _cls_embeddings[2][1], sample_ids),
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.get_bench_dataset_class",
        lambda _: _FakeCls,
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.get_datasets",
        lambda **_: (
            None,
            _DummyLoader(100),
            _DummyLoader(20),
            _DummyLoader(20),
        ),
    )
    monkeypatch.setattr(
        "torchgeo_bench.sample_size_pipeline.instantiate",
        lambda *a, **kw: nn.Identity(),
    )

    from torchgeo_bench.sample_size_pipeline import main as ss_main

    ss_main.__wrapped__(cfg)

    assert len(call_count) == 0
    df = pd.read_csv(csv_path)
    assert len(df) == 48


def test_classification_image_stats_incomplete_block_is_replaced(
    tmp_path,
    monkeypatch,
    _cls_embeddings,
):
    """Resume rewrites an incomplete classification image-stats shard."""
    _patch_cls(monkeypatch, _cls_embeddings)

    import torchgeo_bench.sample_size_pipeline as sample_size_pipeline

    cfg = _make_cls_cfg(
        tmp_path,
        fractions=(0.05,),
        seeds_cls=1,
        resume=True,
        output_name="resume_incomplete.csv",
    )
    meta = sample_size_pipeline._cls_image_stats_block_meta(
        model_name="resnet50",
        model_target="dummy.T",
        dataset_name="m-eurosat",
        partition="default",
        bands="rgb",
        normalization="bandspec_zscore",
        image_size=None,
        interpolation="bilinear",
        train_fraction=0.05,
        seed=0,
        n_train_full=100,
        n_train_used=5,
        n_val=20,
        n_test=20,
    )
    block_key = sample_size_pipeline._image_stats_block_key(meta)
    image_stats_path = sample_size_pipeline._image_stats_block_path(
        root=str(cfg.sample_size.image_stats.root),
        task="classification",
        model="resnet50",
        dataset="m-eurosat",
        train_fraction=0.05,
        seed=0,
        block_key=block_key,
    )
    image_stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                **meta,
                "image_stats_block_key": block_key,
                "best_c": 1.0,
                "image_index": idx,
                "sample_id": f"stale-{idx}",
                "y_true": 0,
                "y_pred": 0,
                "correct": True,
                "confidence": 0.1,
                "margin": 0.1,
                "entropy": 0.1,
                "normalized_entropy": 0.1,
                "nll": 0.1,
            }
            for idx in range(2)
        ]
    ).to_parquet(image_stats_path, index=False)

    sample_size_pipeline.main.__wrapped__(cfg)

    repaired = pd.read_parquet(image_stats_path)
    assert len(repaired) == 20
    assert repaired["sample_id"].iloc[0] == "test-0"
