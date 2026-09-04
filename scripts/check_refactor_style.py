# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Check rewritten modules without reformatting the legacy implementation."""

import subprocess
import sys
from pathlib import Path

PATHS = (
    'scripts/check_refactor_style.py',
    'src/torchgeo_bench/config_schema.py',
    'src/torchgeo_bench/image_cli.py',
    'src/torchgeo_bench/legacy_run.py',
    'src/torchgeo_bench/preprocessing.py',
    'src/torchgeo_bench/models/build.py',
    'src/torchgeo_bench/commands',
    'tests/test_config_schema.py',
    'tests/test_image_cli.py',
    'tests/test_legacy_run.py',
    'tests/test_preprocessing.py',
    'tests/test_model_build.py',
    'tests/test_lazy_commands.py',
    'tests/test_profile_command.py',
)


def main() -> int:
    """Run the same non-mutating checks locally and in pre-commit CI."""
    paths = [path for path in PATHS if Path(path).exists()]
    commands = (
        [
            sys.executable,
            '-m',
            'ruff',
            'format',
            '--check',
            '--config',
            'ruff-refactor.toml',
            *paths,
        ],
        [
            sys.executable,
            '-m',
            'ruff',
            'check',
            '--no-fix',
            '--config',
            'ruff-refactor.toml',
            *paths,
        ],
        [sys.executable, '-m', 'ty', 'check', *paths],
    )
    return max(subprocess.run(command, check=False).returncode for command in commands)


if __name__ == '__main__':
    raise SystemExit(main())
