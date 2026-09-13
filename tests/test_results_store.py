"""Tests for per-model results storage."""

from pathlib import Path

import pandas as pd
import pytest
from omegaconf import OmegaConf

from torchgeo_bench.main import _resolve_output_path
from torchgeo_bench.results import (
    load_results,
    model_results_path,
    sanitize_name,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("timm/vit_base", "timm_vit_base"), ("plain_name", "plain_name"), ("../../model", "model")],
)
def test_sanitize_name_replaces_unsafe_characters(name: str, expected: str) -> None:
    assert sanitize_name(name) == expected


@pytest.mark.parametrize("name", ["///", "..", ""])
def test_sanitize_name_rejects_empty_result(name: str) -> None:
    with pytest.raises(ValueError, match="filename-safe"):
        sanitize_name(name)


def test_model_results_path_sanitizes_names(tmp_path: Path) -> None:
    assert model_results_path(tmp_path, "family/my_model") == tmp_path / "family_my_model.csv"


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


def test_load_results_concatenates_every_model_file(tmp_path: Path) -> None:
    pd.DataFrame([{"name": "a", "metric_value": 1.0}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"name": "b", "metric_value": 2.0}]).to_csv(tmp_path / "b.csv", index=False)
    df = load_results(tmp_path)
    assert len(df) == 2
    assert df.to_dict("records") == [
        {"name": "a", "metric_value": 1.0},
        {"name": "b", "metric_value": 2.0},
    ]
    assert list(df.index) == [0, 1]


def test_load_results_can_select_names(tmp_path: Path) -> None:
    pd.DataFrame([{"name": "a"}]).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame([{"name": "b"}]).to_csv(tmp_path / "b.csv", index=False)
    assert set(load_results(tmp_path, names=["a", "missing"])["name"]) == {"a"}


def test_load_results_empty_directory(tmp_path: Path) -> None:
    assert load_results(tmp_path).empty


def test_load_results_skips_header_only_files(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("name,metric_value\n")
    (tmp_path / "b.csv").write_text("name,metric_value\nb,0.75\n")
    assert load_results(tmp_path).to_dict("records") == [{"name": "b", "metric_value": 0.75}]


def test_load_results_does_not_hide_corrupt_files(tmp_path: Path) -> None:
    (tmp_path / "broken.csv").write_text('name,metric_value\n"unterminated,1\n')
    with pytest.raises(pd.errors.ParserError):
        load_results(tmp_path)
