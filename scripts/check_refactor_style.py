# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Check rewritten modules without reformatting the legacy implementation."""

import ast
import io
import subprocess
import sys
import tokenize
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
    'tests/test_refactor_style.py',
    'tests/test_lazy_commands.py',
)


def check_file(path: Path) -> list[str]:
    """Report oversized docstrings and comment blocks."""
    source = path.read_text(encoding='utf-8')
    errors = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        docstring = ast.get_docstring(node) or ''
        words = len(docstring.split())
        lines = sum(bool(line.strip()) for line in docstring.splitlines())
        if words > 80 or lines > 10:
            errors.append(
                f'{path}:{node.body[0].lineno}: docstring has {words} words and {lines} lines; '
                'limit 80 words and 10 nonblank lines'
            )
    previous = 0
    reported = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type != tokenize.COMMENT or token.line[: token.start[1]].strip():
            continue
        line = token.start[0]
        if line <= 2 and token.string in (
            '# Copyright (c) TorchGeo Contributors. All rights reserved.',
            '# Licensed under the MIT License.',
        ):
            continue
        if line != previous + 1:
            reported = False
        elif previous and not reported:
            errors.append(f'{path}:{previous}: keep comment blocks to one line')
            reported = True
        previous = line
    return errors


def check_prose(paths: list[str]) -> int:
    """Check the same Python files as the refactor gate."""
    files = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files.extend(path.rglob('*.py'))
            files.extend(path.rglob('*.pyi'))
        else:
            files.append(path)
    errors = []
    for path in sorted(set(files)):
        errors.extend(check_file(path))
    for error in errors:
        print(error)
    return int(bool(errors))


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
    status = max(
        subprocess.run(command, check=False).returncode for command in commands
    )
    return status or check_prose(paths)


if __name__ == '__main__':
    raise SystemExit(main())
