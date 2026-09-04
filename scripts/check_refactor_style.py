"""Run the non-mutating style gate for rewritten modules only."""

import subprocess
import sys
from pathlib import Path


PATHS = (
    Path('src/torchgeo_bench/config_schema.py'),
    Path('src/torchgeo_bench/model_profile.py'),
    Path('src/torchgeo_bench/cli.py'),
    Path('src/torchgeo_bench/commands'),
    Path('tests/test_config_schema.py'),
    Path('tests/test_model_profile.py'),
    Path('tests/test_cli.py'),
)

# Existing modules are listed for migration tracking but are deliberately not
# checked until their rewrite lands. This keeps the gate non-mutating and
# prevents a style PR from becoming a legacy-tree formatting PR.
ACTIVE_NAMES = {'config_schema.py', 'test_config_schema.py'}


def main() -> int:
    """Run Ruff and ty against files participating in the rewrite."""
    paths = [str(path) for path in PATHS if path.exists() and path.name in ACTIVE_NAMES]
    if not paths:
        print('No rewritten modules exist yet; style gate is ready.')
        return 0
    commands = (
        [sys.executable, '-m', 'ruff', 'format', '--diff', '--config', 'ruff-refactor.toml', *paths],
        [sys.executable, '-m', 'ruff', 'check', '--no-fix', '--config', 'ruff-refactor.toml', *paths],
    )
    ty_paths = [path for path in paths if path.endswith('.py')]
    if ty_paths:
        commands += ([sys.executable, '-m', 'ty', 'check', *ty_paths],)
    return max(subprocess.run(command, check=False).returncode for command in commands)


if __name__ == '__main__':
    raise SystemExit(main())
