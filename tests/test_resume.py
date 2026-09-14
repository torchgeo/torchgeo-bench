"""Regression tests for resume hashing."""

import pytest

from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.presets import merge_settings, resolve_run_config
from torchgeo_bench.resume import resume_config_hash


def _cfg(**sections) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings({"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}, sections)
    )


def _hash(config: RunConfig, dataset: str = "m-eurosat") -> str:
    resolved, preset = resolve_run_config(config, dataset)
    return resume_config_hash(resolved, preset)


def test_config_hash_ignores_profile_toggle() -> None:
    assert _hash(_cfg()) == _hash(
        _cfg(profile={"enabled": True, "cpu_throughput": {"enabled": True}})
    )


def test_config_hash_ignores_intrinsic_dim_toggle() -> None:
    assert _hash(_cfg()) == _hash(_cfg(intrinsic_dim={"enabled": True}))


def test_dataset_selection_does_not_change_hash() -> None:
    assert _hash(_cfg()) == _hash(_cfg(datasets=["m-forestnet"]))


def test_config_hash_ignores_output_paths_and_method_selection() -> None:
    assert _hash(_cfg()) == _hash(
        _cfg(
            classification={"methods": ["linear"]},
            output={"directory": "other-results", "file": "custom.csv", "resume": True},
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"runtime": {"seed": 1}},
        {"runtime": {"device": "cpu"}},
        {"runtime": {"batch_size": 8}},
        {"runtime": {"workers": 0}},
        {"input": {"bands": "all"}},
        {"input": {"normalization": "minmax"}},
        {"classification": {"knn_k": 7}},
        {"classification": {"knn_device": "cpu"}},
        {"classification": {"linear": {"c_log10_start": -5.0}}},
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
