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
