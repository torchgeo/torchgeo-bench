"""Subprocess coverage of the canonical image CLI entry points."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def run_entrypoint(
    module: str, arguments: list[str], cwd: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.parametrize(
    ("arguments", "status", "expected"),
    [
        (["--help"], 0, "Run image benchmarks"),
        (["run", "--help"], 0, "--config"),
        (["models", "rcf"], 0, "_target_:"),
        (["datasets", "m-eurosat"], 0, "task: classification"),
        ([], 2, "required: command"),
        (["run", "--unknown-option"], 2, "unrecognized arguments: --unknown-option"),
    ],
)
def test_package_module_matches_image_cli(
    arguments: list[str], status: int, expected: str, tmp_path: Path
) -> None:
    package = run_entrypoint("torchgeo_bench", arguments, tmp_path)
    canonical = run_entrypoint("torchgeo_bench.image_cli", arguments, tmp_path)
    assert package.returncode == canonical.returncode == status
    assert package.stdout == canonical.stdout
    assert package.stderr == canonical.stderr
    assert expected in package.stdout + package.stderr


@pytest.mark.parametrize("use_config", [False, True])
def test_package_module_dry_run_matches_image_cli(tmp_path: Path, *, use_config: bool) -> None:
    if use_config:
        path = tmp_path / "run.yaml"
        path.write_text(
            "model: {name: rcf}\ndatasets: [m-eurosat]\n"
            "runtime: {device: cpu, seed: 8}\noutput: {resume: true}\n",
            encoding="utf-8",
        )
        arguments = ["run", "--config", str(path)]
    else:
        arguments = ["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu"]
    arguments.extend(["--seed", "3", "--no-resume", "--dry-run"])
    package = run_entrypoint("torchgeo_bench", arguments, tmp_path)
    canonical = run_entrypoint("torchgeo_bench.image_cli", arguments, tmp_path)
    assert package.returncode == canonical.returncode == 0, package.stderr + canonical.stderr
    assert package.stdout == canonical.stdout
    assert package.stderr == canonical.stderr == ""
    config = yaml.safe_load(package.stdout)
    assert config["model"]["name"] == "rcf"
    assert config["datasets"] == ["m-eurosat"]
    assert config["runtime"] == {"device": "cpu", "seed": 3}
    assert config["output"]["resume"] is False
