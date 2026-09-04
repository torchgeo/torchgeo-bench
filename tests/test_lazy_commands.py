"""Regression tests for lightweight CLI command discovery."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    'arguments',
    [
        ['--help'],
        ['run', '--help'],
        ['run', '--list-models'],
        ['run', '--list-datasets'],
    ],
)
def test_cli_queries_do_not_import_heavy_modules(arguments: list[str]) -> None:
    code = """
import sys
from torchgeo_bench.cli import main
try:
    main(ARGUMENTS)
except SystemExit as error:
    assert error.code in (0, None)
assert not {'torch', 'torchgeo', 'timm', 'pandas'} & sys.modules.keys()
""".replace('ARGUMENTS', repr(arguments))
    environment = {**os.environ, 'PYTHONPATH': 'src'}
    result = subprocess.run([sys.executable, '-c', code], env=environment, check=False)
    assert result.returncode == 0
