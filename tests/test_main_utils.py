"""Unit tests for utility helpers in ``torchgeo_bench.main``."""

from unittest import mock

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from torchgeo_bench.main import (
    _completed_run_keys,
    _expand_dataset_list,
    _filter_completed_metric_rows,
    _normalize_bands_value,
    _resolve_segmentation_runtime_config,
    evaluate_profile,
    resolve_configured_device,
)
from torchgeo_bench.model_profile import measure_cpu_throughput
from torchgeo_bench.segmentation_task import build_seg_probe_and_solver
from torchgeo_bench.settings import SegmentationSettings


class _ImageOnlyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        return {"image": torch.ones(3, 8, 8)}


def test_resolve_configured_device_auto_prefers_cpu_only_when_cuda_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The omitted-device default ("auto") may silently choose CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_configured_device("auto") == torch.device("cpu")


def test_resolve_configured_device_auto_matches_legacy_default_when_cuda_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ "auto" resolves to "cuda:0", matching the string form of the old
    hardcoded default -- so a run that never overrides ``device`` keeps the
    same resume ``config_hash`` it always had."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_configured_device("auto") == torch.device("cuda:0")


def test_resolve_configured_device_explicit_cpu_is_unaffected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_configured_device("cpu") == torch.device("cpu")


def test_expand_dataset_list_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torchgeo_bench.main.list_datasets", lambda: ["m-eurosat", "benv2"])
    assert _expand_dataset_list("all") == ["m-eurosat", "benv2"]


def test_normalize_bands_value_none_and_list() -> None:
    assert _normalize_bands_value(None) == "all"
    assert _normalize_bands_value(["red", "green"]) == "red,green"


def test_completed_run_keys_metric_name_absent_returns_empty() -> None:
    existing = pd.DataFrame([{"dataset": "m-eurosat", "method": "knn5"}])
    assert _completed_run_keys(existing, ["dataset", "method"], metric_name="accuracy") == set()


def test_filter_completed_metric_rows_partial_filtering() -> None:
    rows = [
        {"dataset": "m-eurosat", "method": "knn5", "metric_name": "accuracy"},
        {"dataset": "m-eurosat", "method": "knn5", "metric_name": "f1"},
    ]
    completed = {"accuracy": {("m-eurosat", "knn5")}}
    filtered = _filter_completed_metric_rows(rows, completed, ["dataset", "method"])
    assert filtered == [{"dataset": "m-eurosat", "method": "knn5", "metric_name": "f1"}]


def test_build_seg_probe_and_solver_rejects_empty_layers() -> None:
    segmentation = SegmentationSettings(
        layers=[],
        head_type="fpn",
        criterion={"_target_": "torch.nn.CrossEntropyLoss"},
    )
    with pytest.raises(ValueError, match="requires eval.segmentation.layers"):
        build_seg_probe_and_solver(
            model=torch.nn.Identity(),
            num_classes=2,
            segmentation=segmentation,
            device=torch.device("cpu"),
            lr=1e-3,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("epochs", 0, "positive integer"),
        ("batch_size", 0, "positive integer"),
        ("lr", 0.0, "finite positive number"),
        ("cache_features", "true", "must be a boolean"),
        ("cache_dtype", "bfloat16", "must be one of"),
    ],
)
def test_segmentation_runtime_config_rejects_invalid_values(
    field: str, value: object, message: str
) -> None:
    kwargs = {
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-3,
        "cache_features": True,
        "cache_dtype": "float16",
    }
    kwargs[field] = value
    cfg = SegmentationSettings(**kwargs)

    with pytest.raises(ValueError, match=message):
        _resolve_segmentation_runtime_config(cfg)


def test_measure_cpu_throughput_budget_exceeded_returns_none_metrics() -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, kernel_size=1), torch.nn.ReLU())
    sample = torch.rand(4, 3, 8, 8)
    metrics = measure_cpu_throughput(
        model,
        sample,
        batch_size=2,
        n_warmup=1,
        n_measure=1,
        time_budget_s=0.0,
    )
    assert metrics == {
        "throughput_samples_per_sec_cpu": None,
        "latency_ms_per_batch_p50_cpu": None,
    }


def test_evaluate_profile_adds_cpu_metrics_branch() -> None:
    loader = DataLoader(_ImageOnlyDataset(), batch_size=2, shuffle=False, num_workers=0)
    common_meta = {
        "dataset": "m-eurosat",
        "seed": 0,
        "model": "mock.Model",
        "name": "mock",
        "normalization": "identity",
        "image_size": 8,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "num_classes": 10,
        "config_hash": "test-config-hash",
        "c_range_start": -2,
        "c_range_stop": 2,
        "c_range_num": 3,
        "merge_val": False,
        "bootstrap": 10,
    }

    with (
        mock.patch(
            "torchgeo_bench.model_profile.measure_profile",
            return_value={"params_m": 0.1, "throughput_samples_per_sec": 20.0},
        ),
        mock.patch(
            "torchgeo_bench.model_profile.measure_cpu_throughput",
            return_value={
                "throughput_samples_per_sec_cpu": 3.0,
                "latency_ms_per_batch_p50_cpu": 12.0,
            },
        ),
    ):
        rows = evaluate_profile(
            model=torch.nn.Identity(),
            sample_loader=loader,
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=1,
            common_meta=common_meta,
            feature_dim=8,
            n_counts={"train": 2, "val": 2, "test": 2},
            cpu_throughput_enabled=True,
            cpu_batch_size=2,
            cpu_n_warmup=0,
            cpu_n_measure=1,
            cpu_time_budget_s=1.0,
        )

    metric_names = {row["metric_name"] for row in rows}
    assert "params_m" in metric_names
    assert "throughput_samples_per_sec" in metric_names
    assert "throughput_samples_per_sec_cpu" in metric_names
    assert "latency_ms_per_batch_p50_cpu" in metric_names
