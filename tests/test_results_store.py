"""Tests for per-model results storage."""

import pandas as pd
import pytest
from omegaconf import OmegaConf

from torchgeo_bench.main import _resolve_output_path
from torchgeo_bench.results import (
    DEFAULT_INTRINSIC_DIM_RESULTS_DIR,
    DEFAULT_PROFILE_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
    load_results,
    model_results_path,
    sanitize_name,
)


def test_default_results_dirs_are_distinct():
    dirs = {DEFAULT_RESULTS_DIR, DEFAULT_PROFILE_RESULTS_DIR, DEFAULT_INTRINSIC_DIM_RESULTS_DIR}
    assert len(dirs) == 3
    assert DEFAULT_RESULTS_DIR == "results/models"
    assert DEFAULT_PROFILE_RESULTS_DIR == "results/profiles"
    assert DEFAULT_INTRINSIC_DIM_RESULTS_DIR == "results/intrinsic_dim"


def test_sanitize_name_replaces_unsafe_characters():
    assert sanitize_name("timm/vit_base") == "timm_vit_base"
    assert sanitize_name("plain_name") == "plain_name"


def test_sanitize_name_rejects_empty_result():
    with pytest.raises(ValueError, match="filename-safe"):
        sanitize_name("///")


def test_model_results_path(tmp_path):
    assert model_results_path(tmp_path, "my_model") == tmp_path / "my_model.csv"


def test_resolve_output_path_prefers_explicit_output():
    cfg = OmegaConf.create(
        {"output": "results/scratch.csv", "results_dir": "results/models", "model": {"name": "m"}}
    )
    assert _resolve_output_path(cfg) == "results/scratch.csv"


def test_resolve_output_path_derives_per_model_file():
    cfg = OmegaConf.create(
        {"output": None, "results_dir": "results/models", "model": {"name": "m"}}
    )
    assert _resolve_output_path(cfg) == str(model_results_path("results/models", "m"))


def test_resolve_output_path_requires_a_model_name():
    cfg = OmegaConf.create({"output": None, "results_dir": "results/models", "model": {}})
    with pytest.raises(ValueError, match="no 'name'"):
        _resolve_output_path(cfg)


def test_load_results_concatenates_every_model_file(tmp_path):
    pd.DataFrame([{"name": "a", "metric_value": 1.0}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"name": "b", "metric_value": 2.0}]).to_csv(tmp_path / "b.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 2
    assert set(df["name"]) == {"a", "b"}


def test_load_results_can_select_names(tmp_path):
    pd.DataFrame([{"name": "a"}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"name": "b"}]).to_csv(tmp_path / "b.csv", index=False)
    assert set(load_results(tmp_path, names=["a"])["name"]) == {"a"}


def test_load_results_empty_directory(tmp_path):
    assert load_results(tmp_path).empty


def _dedup_row(**overrides) -> dict:
    row = {
        "dataset": "m-eurosat",
        "method": "linear",
        "model": "timm/resnet50",
        "name": "resnet50",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "num_classes": 10,
        "res": "",
        "pool": "",
        "config_hash": "",
        "metric_name": "accuracy",
        "metric_value": 0.5,
        "seed": 0,
    }
    row.update(overrides)
    return row


def test_load_results_drops_legacy_rows_superseded_by_a_hashed_rerun(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(config_hash="", metric_value=0.5),
            _dedup_row(config_hash="abc123", metric_value=0.6),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 1
    assert df["metric_value"].iloc[0] == pytest.approx(0.6)
    assert df["config_hash"].iloc[0] == "abc123"


def test_load_results_keeps_last_hashed_row_among_hash_vs_hash_ties(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(config_hash="abc123", metric_value=0.6),
            _dedup_row(config_hash="def456", metric_value=0.60001),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 1
    assert df["config_hash"].iloc[0] == "def456"


def test_load_results_keeps_distinct_measurements(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(dataset="m-eurosat", config_hash="abc123"),
            _dedup_row(dataset="m-so2sat", config_hash="abc123"),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    assert len(load_results(tmp_path)) == 2


def test_load_results_keeps_distinct_seeds(tmp_path):
    """Multi-seed sweeps must not collapse to a single arbitrary seed."""
    pd.DataFrame(
        [
            _dedup_row(seed=0, config_hash="a", metric_value=0.81),
            _dedup_row(seed=1, config_hash="b", metric_value=0.83),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 2
    assert sorted(df["seed"]) == [0, 1]


def test_load_results_relabels_olmoearth_normalization(tmp_path):
    pd.DataFrame(
        [
            _dedup_row(name="olmoearth_v1_base", normalization="bandspec_zscore"),
            _dedup_row(name="olmoearth_v1_base_cls", normalization="model_native"),
            _dedup_row(name="tgeo_croma_base", normalization="bandspec_zscore"),
        ]
    ).to_csv(tmp_path / "a.csv", index=False)
    df = load_results(tmp_path).set_index("name")
    assert df.loc["olmoearth_v1_base", "normalization"] == "model_native"
    assert df.loc["olmoearth_v1_base_cls", "normalization"] == "model_native"
    assert df.loc["tgeo_croma_base", "normalization"] == "bandspec_zscore"
