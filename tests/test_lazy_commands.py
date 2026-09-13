"""Regression tests for lightweight CLI command discovery."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    ("module", "arguments"),
    [
        ("torchgeo_bench.cli", ["--help"]),
        ("torchgeo_bench.cli", ["run", "--help"]),
        ("torchgeo_bench.cli", ["run", "--list-models"]),
        ("torchgeo_bench.cli", ["run", "--list-datasets"]),
        ("torchgeo_bench.cli", ["run", "--model-help", "rcf"]),
        ("torchgeo_bench.cli", ["run", "--print-config"]),
        ("torchgeo_bench.cli", ["flops", "--model", "rcf", "--print-config"]),
        ("torchgeo_bench.image_cli", ["--help"]),
        ("torchgeo_bench.image_cli", ["models"]),
        ("torchgeo_bench.image_cli", ["datasets"]),
        ("torchgeo_bench.image_cli", ["run", "--config-help"]),
        ("torchgeo_bench.config_schema", []),
    ],
)
def test_cli_queries_do_not_import_heavy_modules(module: str, arguments: list[str]) -> None:
    code = """
import sys
from importlib import import_module
module = import_module(sys.argv[1])
try:
    if len(sys.argv) > 2:
        module.main(sys.argv[2:])
except SystemExit as error:  # allow-except: help exits successfully without running a benchmark
    assert error.code in (0, None)
assert not {'torch', 'torchgeo', 'timm', 'pandas', 'numpy'} & sys.modules.keys()
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
