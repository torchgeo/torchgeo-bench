"""Process-level entry-point contracts, without importing or faking CLI dispatch."""

from pathlib import Path

import pytest
import yaml

from tests.support.cli import cli_output, run_cli, run_module_cli, run_public_cli

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("module", "commands"),
    [
        ("console", ("run", "models", "datasets", "download", "profile")),
        ("torchgeo_bench.image_cli", ("run", "models", "datasets", "download", "profile")),
        ("torchgeo_bench", ("run", "flops", "profile", "download")),
        ("torchgeo_bench.cli", ("run", "flops", "profile", "download")),
    ],
)
def test_every_entrypoint_and_subcommand_has_help(
    tmp_path: Path, module: str, commands: tuple[str, ...]
) -> None:
    for arguments in (("--help",), *((command, "--help") for command in commands)):
        if module == "console":
            result = run_public_cli(*arguments, cwd=tmp_path)
        else:
            result = run_module_cli(module, *arguments, cwd=tmp_path)
        assert result.returncode == 0, cli_output(result)
        assert "usage:" in result.stdout, cli_output(result)
        assert not result.stderr, cli_output(result)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (("models",), "timm/resnet18"),
        (("models", "rcf"), "_target_: torchgeo_bench.models.RCFBench"),
        (("datasets",), "m-eurosat"),
        (("datasets", "caffe"), "task: segmentation"),
        (("run", "--config-help"), "schema_version"),
    ],
)
def test_public_catalogs_and_schema(
    tmp_path: Path, arguments: tuple[str, ...], expected: str
) -> None:
    result = run_public_cli(*arguments, cwd=tmp_path)
    assert result.returncode == 0, cli_output(result)
    assert expected in result.stdout, cli_output(result)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    ("arguments", "exit_code", "message"),
    [
        ((), 2, "required"),
        (("run", "--model", "rcf"), 2, "datasets"),
        (("models", "missing-preset"), 1, "unknown model"),
        (("download", "missing-dataset"), 1, "Unknown dataset"),
        (("profile", "--model", "rcf"), 2, "dataset"),
    ],
)
def test_public_validation_fails_without_side_effects(
    tmp_path: Path, arguments: tuple[str, ...], exit_code: int, message: str
) -> None:
    result = run_public_cli(*arguments, cwd=tmp_path)
    assert result.returncode == exit_code, cli_output(result)
    assert message in result.stderr, cli_output(result)
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()


def test_public_dry_run_is_reusable_yaml_and_legacy_print_config_is_real(tmp_path: Path) -> None:
    result = run_public_cli(
        "run",
        "--model",
        "rcf",
        "--dataset",
        "m-eurosat",
        "--device",
        "cpu",
        "--image-size",
        "16",
        "--methods",
        "knn",
        "--dry-run",
        cwd=tmp_path,
    )
    assert result.returncode == 0, cli_output(result)
    config = yaml.safe_load(result.stdout)
    assert config["model"] == {"name": "rcf"}
    assert config["datasets"] == ["m-eurosat"]
    assert config["classification"]["methods"] == ["knn"]
    assert config["input"]["image_size"] == 16
    path = tmp_path / "config.yaml"
    path.write_text(result.stdout)
    replay = run_public_cli("run", "--config", str(path), "--dry-run", cwd=tmp_path)
    assert replay.returncode == 0, cli_output(replay)
    assert yaml.safe_load(replay.stdout) == config
    for command in ("run", "flops"):
        legacy = run_cli(command, "model=rcf", "device=cpu", "--print-config", cwd=tmp_path)
        assert legacy.returncode == 0, cli_output(legacy)
        resolved = yaml.safe_load(legacy.stdout)
        assert resolved["model"]["name"] == "rcf"
        assert resolved["device"] == "cpu"
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "results").exists()
