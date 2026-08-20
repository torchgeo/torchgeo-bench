"""Strict configuration and lightweight CLI contracts for supervised priors."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from torchgeo_bench.commands._coord_prior import load_config, run
from torchgeo_bench.commands.coord_prior_arguments import add_coord_prior_arguments
from torchgeo_bench.coordbench.prior_config import CoordPriorConfig, load_coord_prior_config


def _parse(*args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_coord_prior_arguments(parser)
    return parser.parse_args(args)


def test_prior_defaults() -> None:
    config = CoordPriorConfig()
    assert config.datasets == ["all"]
    assert config.evaluation.model_dump() == {
        "methods": ["uniform", "frequency", "grid", "nearest", "kde"],
        "split": "random",
        "folds": 5,
        "cell_deg": 10.0,
        "grid_cell_size": 10.0,
        "smoothing": 0.0,
        "nearest_k": 5,
        "nearest_weights": "uniform",
        "kde_bandwidth": 10.0,
    }
    assert config.runtime.model_dump() == {"seed": 0}
    assert config.output.model_dump() == {"file": "results/coordbench_priors.csv", "resume": False}


@pytest.mark.parametrize(
    "values",
    [
        {"unexpected": True},
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": 2},
        {"model": {"name": "sincos"}},
        {"datasets": "country"},
        {"datasets": []},
        {"datasets": [2]},
        {"datasets": [" "]},
        {"datasets": ["country", "country"]},
        {"datasets": ["all", "country"]},
        {"datasets": ["missing"]},
        {"evaluation": {"methods": []}},
        {"evaluation": {"methods": ["grid", "grid"]}},
        {"evaluation": {"methods": ["knn"]}},
        {"evaluation": {"split": "official"}},
        {"evaluation": {"folds": 1}},
        {"evaluation": {"folds": "5"}},
        {"evaluation": {"folds": True}},
        {"evaluation": {"cell_deg": 0}},
        {"evaluation": {"cell_deg": "10.0"}},
        {"evaluation": {"cell_deg": float("nan")}},
        {"evaluation": {"grid_cell_size": -1}},
        {"evaluation": {"grid_cell_size": True}},
        {"evaluation": {"grid_cell_size": float("inf")}},
        {"evaluation": {"smoothing": -1}},
        {"evaluation": {"smoothing": "0"}},
        {"evaluation": {"smoothing": float("nan")}},
        {"evaluation": {"nearest_k": 0}},
        {"evaluation": {"nearest_k": 1.5}},
        {"evaluation": {"nearest_k": True}},
        {"evaluation": {"nearest_weights": "inverse"}},
        {"evaluation": {"kde_bandwidth": 0}},
        {"evaluation": {"kde_bandwidth": "10"}},
        {"evaluation": {"kde_bandwidth": float("inf")}},
        {"evaluation": {"knn_device": "cpu"}},
        {"runtime": {"device": "cpu"}},
        {"runtime": {"seed": -1}},
        {"runtime": {"seed": 2**64}},
        {"runtime": {"seed": True}},
        {"runtime": {"seed": 0.5}},
        {"output": {"file": None}},
        {"output": {"file": " "}},
        {"output": {"resume": "false"}},
        {"output": {"resume": 0}},
        {"output": {"directory": "results"}},
    ],
)
def test_prior_rejects_invalid_config(values: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CoordPriorConfig.model_validate(values)


@pytest.mark.parametrize("name", ["all", "country", "satclip", "satclip-country", "dm-descals"])
@pytest.mark.parametrize("flag", ["--dataset", "--datasets", "--name", "--names"])
def test_prior_accepts_catalog_names_and_flag_aliases(name: str, flag: str) -> None:
    assert load_config(_parse(flag, name)).datasets == [name]


def test_prior_flags_override_yaml_without_overriding_omitted_settings(tmp_path: Path) -> None:
    path = tmp_path / "prior.yaml"
    path.write_text(
        "datasets: [country]\nevaluation:\n  methods: [nearest]\n  split: spatial\n"
        "  folds: 8\n  cell_deg: 20.0\n  grid_cell_size: 30.0\n  smoothing: 1.0\n"
        "  nearest_k: 9\n  nearest_weights: distance\n  kde_bandwidth: 1e1\n"
        "runtime: {seed: 17}\noutput: {file: custom.csv, resume: true}\n"
    )
    config = load_config(
        _parse(
            "--config",
            str(path),
            "--seed",
            "0",
            "--no-resume",
            "--smoothing",
            "0",
            "--grid-cell-size",
            "5",
            "--nearest-k",
            "3",
            "--nearest-weights",
            "uniform",
            "--kde-bandwidth",
            "4",
            "--cell-deg",
            "2",
            "--folds",
            "3",
            "--split",
            "both",
            "--dataset",
            "satclip-country",
            "--methods",
            "grid",
            "kde",
        )
    )
    assert config.datasets == ["satclip-country"]
    assert config.evaluation.model_dump() == {
        "methods": ["grid", "kde"],
        "split": "both",
        "folds": 3,
        "cell_deg": 2.0,
        "grid_cell_size": 5.0,
        "smoothing": 0.0,
        "nearest_k": 3,
        "nearest_weights": "uniform",
        "kde_bandwidth": 4.0,
    }
    assert config.runtime.seed == 0
    assert config.output.file == "custom.csv"
    assert not config.output.resume
    original = load_coord_prior_config(path)
    assert original.output.resume
    assert original.evaluation.kde_bandwidth == 10.0
    assert load_config(_parse("--config", str(path))) == original


@pytest.mark.parametrize(
    "content",
    [
        "[]",
        "datasets: [",
        "datasets: [country]\ndatasets: [satclip]",
        "runtime: !!python/object:a {}",
        "evaluation: null",
        "runtime: {seed: false}",
    ],
)
def test_bad_prior_yaml_reports_argparse_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(SystemExit) as error:
        run(_parse("--config", str(path), "--dry-run"))
    assert error.value.code == 2
    assert str(path) in capsys.readouterr().err


def test_missing_prior_yaml_reports_argparse_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "absent.yaml"
    with pytest.raises(SystemExit) as error:
        run(_parse("--config", str(path)))
    assert error.value.code == 2
    assert str(path) in capsys.readouterr().err


def test_prior_dry_run_roundtrips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(_parse("--dataset", "country", "--methods", "frequency", "--dry-run"))
    path = tmp_path / "prior.yaml"
    path.write_text(capsys.readouterr().out)
    config = load_coord_prior_config(path)
    assert config.evaluation.methods == ["frequency"]
    assert load_config(_parse("--config", str(path))) == config


@pytest.mark.parametrize("arguments", [["coord-prior", "--help"], ["coord-prior", "--dry-run"]])
def test_prior_help_and_dry_run_do_not_import_runtime(arguments: list[str]) -> None:
    script = """
import importlib.abc
import sys

class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {
            "torch", "numpy", "pandas", "omegaconf", "sklearn", "huggingface_hub"
        } or fullname in {
            "torchgeo_bench.coordbench.prior_run", "torchgeo_bench.coordbench.datasets",
            "torchgeo_bench.coordbench.baselines", "torchgeo_bench.coordbench.models",
            "torchgeo_bench.main",
        }:
            raise AssertionError(f"lightweight command imported {fullname}")

sys.meta_path.insert(0, BlockRuntime())
from torchgeo_bench.cli import main
main(sys.argv[1:])
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0, result.stderr


def test_prior_command_dispatches_typed_config(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchgeo_bench.cli import main

    configs = []
    monkeypatch.setattr("torchgeo_bench.coordbench.prior_run.run_coordbench_priors", configs.append)
    main(["coord-prior", "--dataset", "country", "--methods", "frequency"])
    assert len(configs) == 1
    assert isinstance(configs[0], CoordPriorConfig)
    assert configs[0].datasets == ["country"]


def test_prior_documented_example_is_valid() -> None:
    path = Path(__file__).resolve().parents[1] / "docs/examples/coord-prior.yaml"
    assert load_coord_prior_config(path).datasets == ["satclip-country"]
