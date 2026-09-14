"""Regression tests for lightweight CLI command discovery."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["run", "--help"],
        ["run", "--config-help"],
        ["models"],
        ["datasets"],
        ["models", "rcf"],
        ["datasets", "m-eurosat"],
        ["datasets", "burn_scars"],
        ["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run"],
        ["download", "--help"],
        ["profile", "--help"],
        ["profile", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run"],
        ["flops", "--help"],
        ["flops", "--config-help"],
        ["flops", "--model", "rcf", "--dry-run"],
        ["coord", "--help"],
        ["coord", "--model", "sincos", "--dataset", "california_housing", "--dry-run"],
    ],
)
def test_cli_queries_do_not_import_heavy_modules(arguments: list[str]) -> None:
    code = """
import sys
blocked = {'torch', 'torchgeo', 'timm', 'numpy', 'pandas', 'omegaconf'}
class BlockRuntime:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError(f'Unexpected runtime import: {fullname}')
sys.meta_path.insert(0, BlockRuntime())
from torchgeo_bench.cli import main
try:
    main(ARGUMENTS)
except SystemExit as error:  # allow-except: help exits successfully without running a benchmark
    assert error.code in (0, None)
assert not blocked & sys.modules.keys()
""".replace("ARGUMENTS", repr(arguments))
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout


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
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert result.stdout == ""
    assert result.stderr == "WARNING: sample message\n"
