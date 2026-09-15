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
        ("torchgeo_bench.config_schema", []),
        ("torchgeo_bench.run_config", []),
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
