"""Regression tests for resume hashing and metric completeness."""

from pathlib import Path

import pandas as pd
import pytest

from tests.support.runner import _resume_row
from torchgeo_bench.config.presets import merge_settings, resolve_run_config
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.resume import (
    ResumeState,
    load_completed,
    plan_dataset_run,
    resume_config_hash,
)


def _cfg(**sections) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings({"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}, sections)
    )


def _hash(config: RunConfig, dataset: str = "m-eurosat") -> str:
    resolved, preset = resolve_run_config(config, dataset)
    return resume_config_hash(resolved, preset)


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": {"enabled": True, "cpu_throughput": {"enabled": True}}},
        {"intrinsic_dim": {"enabled": True}},
        {"datasets": ["m-forestnet"]},
        {"output": {"resume": True}},
    ],
    ids=["profile-pass", "intrinsic-dim-pass", "dataset-selection", "resume-toggle"],
)
def test_config_hash_ignores_run_selection_and_additive_passes(overrides: dict) -> None:
    assert _hash(_cfg()) == _hash(_cfg(**overrides))


def test_config_hash_ignores_output_paths_and_method_selection() -> None:
    assert _hash(_cfg()) == _hash(
        _cfg(
            classification={"methods": ["linear"]},
            output={"directory": "other-results", "file": "custom.csv", "resume": True},
        )
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:7"])
@pytest.mark.parametrize("workers", [0, 4, 8])
def test_config_hash_ignores_device_and_workers(device: str, workers: int) -> None:
    assert _hash(_cfg()) == _hash(_cfg(runtime={"device": device, "workers": workers}))


@pytest.mark.parametrize("runtime", [{"seed": 1}, {"batch_size": 8}])
def test_config_hash_retains_seed_and_batch_size(runtime: dict[str, int]) -> None:
    model = {"name": "custom", "target": "example.Model"}
    assert _hash(_cfg(model=model)) != _hash(_cfg(model=model, runtime=runtime))


@pytest.mark.parametrize(
    "overrides",
    [
        {"input": {"bands": "all"}},
        {"input": {"normalization": "minmax"}},
        {"classification": {"knn_k": 7}},
        {"classification": {"knn_device": "cpu"}},
        {"classification": {"linear": {"c_log10_start": -5.0}}},
        {"classification": {"linear": {"refit_train_val": False}}},
        {"classification": {"calibration": {"n_bins_linear": 10}}},
        {"classification": {"bootstrap_samples": 100}},
        {"segmentation": {"head": "linear"}},
        {"segmentation": {"learning_rate": 0.01}},
        {"model": {"name": "rcf", "kwargs": {"mode": "empirical"}}},
    ],
)
def test_config_hash_changes_with_result_affecting_settings(overrides: dict) -> None:
    assert _hash(_cfg()) != _hash(_cfg(**overrides))


def test_config_hash_includes_resolved_model_target_and_kwargs() -> None:
    custom = _cfg(model={"name": "custom", "target": "example.Model", "kwargs": {"width": 4}})
    wider = _cfg(model={"name": "custom", "target": "example.Model", "kwargs": {"width": 8}})
    other_target = _cfg(model={"name": "custom", "target": "example.Other", "kwargs": {"width": 4}})

    assert _hash(custom) != _hash(wider)
    assert _hash(custom) != _hash(other_target)


def test_config_hash_reflects_dataset_resolved_preset_overrides() -> None:
    config = _cfg(model={"name": "torchgeo/scalemae_large_fmow"})
    assert _hash(config, "m-eurosat") != _hash(config, "forestnet")


def test_resume_keys_require_the_requested_metric(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [{"dataset": "m-eurosat", "method": "intrinsic_dim", "metric_name": "id_twonn_train"}]
    ).to_csv(path, index=False)
    columns = ("dataset", "method")
    key = ("m-eurosat", "intrinsic_dim")
    completed, metrics = load_completed(str(path), columns)
    assert completed == {key}
    assert metrics["id_twonn_train"] == {key}
    assert metrics.get("id_mle_train", set()) == set()


@pytest.mark.parametrize("stored_hash", ["0391f898e8a4db0d", "21d7c33e4e3fb14b", ""])
def test_resume_rejects_legacy_runtime_hashes(tmp_path: Path, stored_hash: str) -> None:
    # Pre-#401 hashes for default RCF / m-eurosat on cuda:0 and cpu, plus an unhashed row.
    config = _cfg(classification={"methods": ["knn"]}, output={"resume": True})
    metadata = _resume_row(config, method="knn5", metric_name="accuracy")
    path = tmp_path / "results.csv"
    pd.DataFrame([{**metadata, "config_hash": stored_hash}]).to_csv(path, index=False)
    completed = ResumeState(*load_completed(str(path)))
    dataset = get_bench_dataset_class("m-eurosat")
    plan = plan_dataset_run(config, dataset, metadata, completed)
    assert not plan.skip_dataset
    assert not plan.skip_knn


def test_runtime_change_keeps_missing_additive_passes_pending(tmp_path: Path) -> None:
    config = _cfg(classification={"methods": ["knn"]}, output={"resume": True})
    path = tmp_path / "results.csv"
    pd.DataFrame([_resume_row(config, method="knn5", metric_name="accuracy")]).to_csv(
        path, index=False
    )
    completed = ResumeState(*load_completed(str(path)))
    config.runtime.device = "cpu"
    config.runtime.workers = 8
    config.profile.enabled = True
    config.intrinsic_dim.enabled = True
    metadata = _resume_row(config, method="knn5", metric_name="accuracy")
    plan = plan_dataset_run(config, get_bench_dataset_class("m-eurosat"), metadata, completed)
    assert plan.skip_knn
    assert plan.skip_linear
    assert not plan.skip_dataset
    assert not plan.skip_profile
    assert not plan.skip_id
    assert plan.id_missing_metrics


@pytest.mark.parametrize("cpu_enabled", [False, True])
def test_profile_resume_requires_every_enabled_metric(tmp_path: Path, *, cpu_enabled: bool) -> None:
    config, _ = resolve_run_config(
        _cfg(
            output={"resume": True},
            profile={"enabled": True, "cpu_throughput": {"enabled": cpu_enabled}},
        ),
        "m-eurosat",
    )
    expected = {"throughput_samples_per_sec", "latency_ms_per_batch_p50", "params_m"}
    if cpu_enabled:
        expected |= {"throughput_samples_per_sec_cpu", "latency_ms_per_batch_p50_cpu"}
    path = tmp_path / "profile.csv"
    rows = [_resume_row(config, method="profile", metric_name=name) for name in sorted(expected)]
    pd.DataFrame(rows).to_csv(path, index=False)
    completed = ResumeState(*load_completed(str(path)))
    config.runtime.device = "cpu"
    config.runtime.workers = 8
    metadata = _resume_row(config, method="profile", metric_name="params_m")
    dataset = get_bench_dataset_class("m-eurosat")
    assert plan_dataset_run(config, dataset, metadata, completed).skip_profile
    for name in expected:
        partial = ResumeState(
            completed.completed_runs,
            {
                metric: keys
                for metric, keys in completed.completed_metrics.items()
                if metric != name
            },
        )
        assert not plan_dataset_run(config, dataset, metadata, partial).skip_profile


def test_load_completed_canonicalizes_csv_numbers_and_legacy_columns(tmp_path: Path) -> None:
    path = tmp_path / "results.csv"
    path.write_text(
        "dataset,image_size,metric_name\n"
        "m-eurosat,224.0,accuracy\n"
        "m-eurosat,224,accuracy\n"
        "m-forestnet,,micro_mAP\n"
    )
    completed, metrics = load_completed(str(path), ("dataset", "image_size", "config_hash"))
    assert completed == {("m-eurosat", "224", ""), ("m-forestnet", "", "")}
    assert metrics == {
        "accuracy": {("m-eurosat", "224", "")},
        "micro_mAP": {("m-forestnet", "", "")},
    }
