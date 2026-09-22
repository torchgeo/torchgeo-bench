"""Regression tests for lightweight CLI command discovery."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        ("torchgeo_bench.cli", ["--help"]),
        ("torchgeo_bench.cli", ["--version"]),
        ("torchgeo_bench.cli", ["run", "--help"]),
        ("torchgeo_bench.cli", ["run", "--config-help"]),
        ("torchgeo_bench.cli", ["models"]),
        ("torchgeo_bench.cli", ["datasets"]),
        ("torchgeo_bench.cli", ["models", "rcf"]),
        ("torchgeo_bench.cli", ["datasets", "m-eurosat"]),
        ("torchgeo_bench.cli", ["datasets", "burn_scars"]),
        (
            "torchgeo_bench.cli",
            ["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run"],
        ),
        ("torchgeo_bench.cli", ["download", "--help"]),
        ("torchgeo_bench.cli", ["profile", "--help"]),
        (
            "torchgeo_bench.cli",
            ["profile", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run"],
        ),
        ("torchgeo_bench.cli", ["flops", "--help"]),
        ("torchgeo_bench.cli", ["flops", "--config-help"]),
        ("torchgeo_bench.cli", ["flops", "--model", "rcf", "--dry-run"]),
        ("torchgeo_bench.cli", ["coord", "--help"]),
        (
            "torchgeo_bench.cli",
            ["coord", "--model", "sincos", "--dataset", "california_housing", "--dry-run"],
        ),
        ("torchgeo_bench.config", []),
        ("torchgeo_bench.config.catalog", []),
        ("torchgeo_bench.config.schema", []),
        ("torchgeo_bench.config.presets", []),
        ("torchgeo_bench.config.profile", []),
        ("torchgeo_bench.config.flops", []),
        ("torchgeo_bench.config.run", []),
    ],
)
def test_cli_queries_do_not_import_heavy_modules(module: str, arguments: list[str]) -> None:
    code = """
import sys
from importlib import import_module
blocked = {'torch', 'torchgeo', 'timm', 'numpy', 'pandas', 'omegaconf'}
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError(f'Unexpected runtime import: {fullname}')
sys.meta_path.insert(0, BlockRuntime())
module = import_module(sys.argv[1])
try:
    if len(sys.argv) > 2:
        module.main(sys.argv[2:])
except SystemExit as error:  # allow-except: help exits without running a benchmark
    assert error.code in (0, None)
assert not blocked & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, module, *arguments],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_catalog_does_not_import_schemas() -> None:
    code = """
import sys
from torchgeo_bench.config import list_model_configs, model_config_path
assert "rcf" in list_model_configs()
assert model_config_path("rcf").is_file()
assert not {"pydantic", "yaml"} & sys.modules.keys()
assert not {
    "torchgeo_bench.config.schema",
    "torchgeo_bench.config.presets",
    "torchgeo_bench.config.profile",
    "torchgeo_bench.config.flops",
    "torchgeo_bench.config.run",
} & sys.modules.keys()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def test_command_logging_uses_standard_logging() -> None:
    code = """
import logging
import sys
from torchgeo_bench.commands._common import setup_logging
setup_logging(verbose=True)
logging.getLogger("test").warning("sample message")
assert "rich" not in sys.modules
assert "torch" not in sys.modules
assert "omegaconf" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    assert result.stdout == ""
    assert result.stderr == "WARNING: sample message\n"


@pytest.mark.parametrize(
    ("model", "flags", "status"),
    [
        ("rcf", ["--bands", "red,green,blue"], 0),
        ("rcf", ["--bands", "B99"], 2),
        ("rcf", ["--normalization", "model"], 2),
        ("olmoearth_nano", ["--normalization", "model"], 0),
        ("terratorch/clay_v1_5", ["--bands", "red,green,blue"], 0),
    ],
)
def test_run_metadata_validation_loads_no_models_or_samples(
    model: str, flags: list[str], status: int
) -> None:
    code = """
import sys
blocked = {'olmoearth_pretrain_minimal', 'terratorch', 'rshf'}
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if (fullname.split('.')[0] in blocked
            or fullname.startswith('torchgeo_bench.models')
            or fullname == 'torchgeo_bench.commands._run_runtime'):
            raise AssertionError(f'Unexpected runtime import: {fullname}')
sys.meta_path.insert(0, BlockRuntime())
from torchgeo_bench.cli import main
from torchgeo_bench.datasets.base import BenchDataset
def no_dataset(*args, **kwargs):
    raise AssertionError('Metadata validation must not construct datasets')
BenchDataset.__init__ = no_dataset
status = 0
try:
    main(sys.argv[2:])
except SystemExit as error:  # allow-except: inspect expected CLI validation exits
    status = error.code
assert status == int(sys.argv[1])
assert not blocked & sys.modules.keys()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(status),
            "run",
            "--model",
            model,
            "--dataset",
            "m-eurosat",
            *flags,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
