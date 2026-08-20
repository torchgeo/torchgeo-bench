"""Offline prior evaluation through the installed and both module entry points."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.support.cli import cli_output, run_module_cli, run_public_cli

pytestmark = pytest.mark.integration


def _run_entrypoint(
    module: str, arguments: tuple[str, ...], cwd: Path
) -> subprocess.CompletedProcess[str]:
    if module == "console":
        return run_public_cli(*arguments, cwd=cwd)
    return run_module_cli(module, *arguments, cwd=cwd)


@pytest.mark.parametrize("module", ["console", "torchgeo_bench", "torchgeo_bench.cli"])
def test_prior_real_dispatch_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    cache = tmp_path / "hub"
    repo = cache / "datasets--taylor-geospatial--coordbench"
    revision = "0" * 40
    (repo / "refs").mkdir(parents=True)
    (repo / "refs/main").write_text(revision)
    table = repo / "snapshots" / revision / "data/country/data.parquet"
    table.parent.mkdir(parents=True)
    lat = np.linspace(20.0, 50.0, 30)
    pd.DataFrame(
        {"lon": np.linspace(-120.0, -80.0, 30), "lat": lat, "country": (lat > 35.0).astype(int)}
    ).to_parquet(table)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    config = tmp_path / "prior.yaml"
    config.write_text(
        "datasets: [country]\nevaluation: {split: both, folds: 3}\n"
        "runtime: {seed: 9}\noutput: {file: priors.csv, resume: true}\n"
    )
    args = ("coord-prior", "--config", str(config), "--seed", "0")
    result = _run_entrypoint(module, args, tmp_path)
    assert result.returncode == 0, cli_output(result)
    output = tmp_path / "priors.csv"
    original = pd.read_csv(output)
    assert len(original) == 10
    assert set(original.model_name) == {"uniform", "frequency", "grid", "nearest", "kde"}
    assert set(original.method) == {"prior"}
    assert set(original.split) == {"random", "spatial"}
    assert (original.n_test == 30).all()
    assert (original.seed == 0).all()
    assert np.isfinite(original.metric_value).all()
    result = _run_entrypoint(module, args, tmp_path)
    assert result.returncode == 0, cli_output(result)
    pd.testing.assert_frame_equal(pd.read_csv(output), original)
    result = _run_entrypoint(module, (*args, "--no-resume"), tmp_path)
    assert result.returncode == 0, cli_output(result)
    assert len(pd.read_csv(output)) == 20


@pytest.mark.parametrize("module", ["console", "torchgeo_bench", "torchgeo_bench.cli"])
def test_prior_entrypoints_dry_run_and_help(tmp_path: Path, module: str) -> None:
    result = _run_entrypoint(module, ("coord-prior", "--help"), tmp_path)
    assert result.returncode == 0, cli_output(result)
    assert "--nearest-weights" in result.stdout
    assert "--device" not in result.stdout
    result = _run_entrypoint(
        module, ("coord-prior", "--dataset", "country", "--methods", "grid", "--dry-run"), tmp_path
    )
    assert result.returncode == 0, cli_output(result)
    assert yaml.safe_load(result.stdout)["evaluation"]["methods"] == ["grid"]
    assert not (tmp_path / "results").exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("module", ["console", "torchgeo_bench", "torchgeo_bench.cli"])
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--device", "cuda"), "unrecognized arguments"),
        (("--model", "sincos"), "unrecognized arguments"),
        (("--folds", "1"), "greater than or equal to 2"),
        (("--dataset", "missing"), "Unknown coordinate datasets"),
        (("--smoothing", "-1"), "greater than or equal to 0"),
        (("--methods", "knn"), "invalid choice"),
        (("mode=coord-prior",), "key=value and +key=value overrides have been retired"),
    ],
)
def test_prior_entrypoint_errors_have_status_two(
    tmp_path: Path, module: str, arguments: tuple[str, ...], message: str
) -> None:
    result = _run_entrypoint(module, ("coord-prior", *arguments), tmp_path)
    assert result.returncode == 2, cli_output(result)
    assert message in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results").exists()
