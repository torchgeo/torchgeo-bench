"""Regression tests for resume-mode config fingerprinting."""

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from torchgeo_bench.config import compose_config
from torchgeo_bench.resume import (
    _completed_run_keys,
    _profile_metric_names,
    _resume_config_hash,
    load_completed,
)


def _cfg(overrides: list[str]) -> DictConfig:
    return compose_config(["model=rcf", "dataset.names=[m-eurosat]", *overrides])


@pytest.mark.parametrize(
    "overrides",
    [
        ["eval.profile.enabled=true", "eval.profile.cpu_throughput.enabled=true"],
        ["eval.intrinsic_dim.enabled=true"],
        ["dataset.names=[m-forestnet]"],
        ["resume=true"],
    ],
    ids=["profile-pass", "intrinsic-dim-pass", "dataset-selection", "resume-toggle"],
)
def test_config_hash_ignores_run_selection_and_additive_passes(overrides: list[str]) -> None:
    assert _resume_config_hash(_cfg([])) == _resume_config_hash(_cfg(overrides))


@pytest.mark.parametrize(
    "override",
    ["dataset.normalization=minmax", "seed=1", "eval.merge_val=false", "eval.bootstrap=11"],
)
def test_config_hash_changes_with_result_affecting_settings(override: str) -> None:
    assert _resume_config_hash(_cfg([])) != _resume_config_hash(_cfg([override]))


def test_removed_plot_defaults_keep_existing_resume_keys() -> None:
    current = _cfg([])
    previous = _cfg(
        [
            "+eval.segmentation.save_viz=false",
            "+eval.segmentation.viz_dir=viz",
            "+eval.segmentation.n_viz_samples=8",
        ]
    )
    assert _resume_config_hash(current) == _resume_config_hash(previous)
    assert not {"save_viz", "viz_dir", "n_viz_samples"} & set(current.eval.segmentation)


def test_resume_keys_require_the_requested_metric() -> None:
    df = pd.DataFrame(
        [{"dataset": "m-eurosat", "method": "intrinsic_dim", "metric_name": "id_twonn_train"}]
    )
    columns = ("dataset", "method")
    key = ("m-eurosat", "intrinsic_dim")
    assert _completed_run_keys(df, columns, "id_twonn_train") == {key}
    assert _completed_run_keys(df, columns, "id_mle_train") == set()


@pytest.mark.parametrize("cpu_enabled", [False, True])
def test_profile_resume_requires_every_enabled_metric(*, cpu_enabled: bool) -> None:
    metrics = _profile_metric_names(OmegaConf.create({"cpu_throughput": {"enabled": cpu_enabled}}))
    expected = {"throughput_samples_per_sec", "latency_ms_per_batch_p50", "params_m"}
    if cpu_enabled:
        expected |= {"throughput_samples_per_sec_cpu", "latency_ms_per_batch_p50_cpu"}
    assert set(metrics) == expected


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
