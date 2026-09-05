# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for prose limits in rewritten Python files."""

import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts' / 'check_refactor_style.py'
spec = importlib.util.spec_from_file_location('refactor_style', SCRIPT)
assert spec is not None
assert spec.loader is not None
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        ('# One reason.\nvalue = 1\n# Another reason.\n', 0),
        ('# One reason.\n# More background.\n# Even more background.\n', 1),
        ('# One reason.\n\n# Another reason.\n', 0),
        ('value = 1  # One reason.\nother = 2  # Another reason.\n', 0),
        ('value = """# This is data.\n# Still data.\n"""\n', 0),
        (
            '# Copyright (c) TorchGeo Contributors. All rights reserved.\n'
            '# Licensed under the MIT License.\n',
            0,
        ),
    ],
)
def test_comment_blocks(tmp_path: Path, source: str, expected: int) -> None:
    path = tmp_path / 'example.py'
    path.write_text(source, encoding='utf-8')
    errors = style.check_file(path)
    assert len(errors) == expected
    if expected:
        assert errors == [f'{path}:1: keep comment blocks to one line']


@pytest.mark.parametrize(
    'prefix', ['', 'class Model:\n', 'def run():\n', 'async def run():\n']
)
@pytest.mark.parametrize(('words', 'expected'), [(80, 0), (81, 1)])
def test_docstring_word_limit(
    tmp_path: Path, prefix: str, words: int, expected: int
) -> None:
    path = tmp_path / 'example.py'
    indent = '    ' if prefix else ''
    path.write_text(
        prefix + indent + repr(' '.join(['word'] * words)), encoding='utf-8'
    )
    errors = style.check_file(path)
    assert len(errors) == expected
    if expected:
        assert '81 words and 1 lines' in errors[0]


@pytest.mark.parametrize(('lines', 'expected'), [(10, 0), (11, 1)])
def test_docstring_line_limit(tmp_path: Path, lines: int, expected: int) -> None:
    path = tmp_path / 'example.py'
    path.write_text(repr('\n\n'.join(['word'] * lines)), encoding='utf-8')
    errors = style.check_file(path)
    assert len(errors) == expected
    if expected:
        assert '11 words and 11 lines' in errors[0]


def test_recursive_paths_are_deduplicated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / 'example.py'
    path.write_text('# One reason.\n# More background.\n', encoding='utf-8')
    (tmp_path / 'data.txt').write_text('not Python', encoding='utf-8')
    assert style.check_prose([str(tmp_path), str(path)]) == 1
    assert capsys.readouterr().out == f'{path}:1: keep comment blocks to one line\n'


@pytest.mark.parametrize('returncode', [0, 1])
def test_gate_runs_tools_and_prose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    path = tmp_path / 'example.py'
    path.write_text('value = 1\n', encoding='utf-8')
    monkeypatch.setattr(style, 'PATHS', (str(path), str(tmp_path / 'missing.py')))
    calls = []

    def run(command: list[str], *, check: bool) -> SimpleNamespace:
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(style.subprocess, 'run', run)
    assert style.main() == returncode
    assert len(calls) == 3
    assert all(command[-1] == str(path) for command in calls)


def test_script_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        style.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0)
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name='__main__')
    assert error.value.code == 0
