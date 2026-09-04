# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the discoverable image CLI."""

import runpy
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import DictConfig, open_dict

from torchgeo_bench.config_schema import validate_run_config
from torchgeo_bench.image_cli import _image_size, _model_names, _set, main


def test_dry_run_applies_explicit_flags_and_preserves_false_values(
    capsys: CaptureFixture[str],
) -> None:
    main(
        [
            'run',
            '--model',
            'rcf',
            '--dataset',
            'm-eurosat',
            '--dataset',
            'burn_scars',
            '--image-size',
            'none',
            '--no-resume',
            '--methods',
            'knn',
            '--dry-run',
        ]
    )
    output = capsys.readouterr().out
    assert 'image_size: null' in output
    assert 'resume: false' in output
    assert '- burn_scars' in output
    assert 'methods:\n  - knn' in output


def test_config_values_are_overridden_by_explicit_flags(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        'model: {name: rcf}\ndatasets: [m-eurosat]\nruntime: {seed: 8}\noutput: {resume: true}\n',
        encoding='utf-8',
    )
    main(['run', '--config', str(path), '--seed', '3', '--no-resume', '--dry-run'])
    output = capsys.readouterr().out
    assert 'seed: 3' in output
    assert 'resume: false' in output


def test_nested_flag_mapping_and_image_size_validation() -> None:
    mapping = {}
    _set(mapping, 'classification.linear', 'refit_train_val', False)
    assert mapping == {'classification': {'linear': {'refit_train_val': False}}}
    assert _image_size('none') is None
    assert _image_size('224') == 224
    with pytest.raises(Exception, match='positive'):
        _image_size('0')


def test_boolean_flags_override_config(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        'model: {name: rcf}\ndatasets: [m-eurosat]\n'
        'classification:\n  linear:\n    refit_train_val: true\n',
        encoding='utf-8',
    )
    main(
        [
            'run',
            '--config',
            str(path),
            '--no-refit-train-val',
            '--no-temp-scale',
            '--dry-run',
        ]
    )
    assert 'refit_train_val: false' in capsys.readouterr().out


def test_missing_model_or_dataset_fails_before_execution() -> None:
    with pytest.raises(SystemExit, match='2'):
        main(['run', '--dataset', 'm-eurosat', '--dry-run'])
    with pytest.raises(SystemExit, match='2'):
        main(['run', '--model', 'rcf', '--dry-run'])


def test_non_dry_run_calls_legacy_adapter(monkeypatch: MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr('torchgeo_bench.legacy_run.run', received.append)
    main(['run', '--model', 'rcf', '--dataset', 'm-eurosat'])
    assert received[0].model.name == 'rcf'


def test_legacy_adapter_composes_and_translates_schema(
    monkeypatch: MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr('torchgeo_bench.main.main', capture)
    config = validate_run_config(
        {
            'model': {'name': 'rcf'},
            'datasets': ['m-eurosat'],
            'runtime': {'device': 'cpu', 'batch_size': 2},
            'input': {
                'normalization': 'none',
                'image_size': 8,
                'time_steps': 1,
                'interpolation': 'nearest',
            },
            'classification': {'methods': ['knn']},
        }
    )
    from torchgeo_bench.legacy_run import run

    run(config)
    assert received[0].device == 'cpu'
    assert received[0].dataset.normalization == 'identity'
    assert received[0].eval.skip_linear is True


def test_legacy_adapter_rejects_unavailable_cuda(monkeypatch: MonkeyPatch) -> None:
    import torch

    config = validate_run_config(
        {
            'model': {'name': 'rcf'},
            'datasets': ['m-eurosat'],
            'runtime': {'device': 'cuda:0'},
        }
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    from torchgeo_bench.legacy_run import run

    with pytest.raises(RuntimeError, match='CUDA device'):
        run(config)


def test_legacy_adapter_preserves_model_and_segmentation_overrides(
    monkeypatch: MonkeyPatch,
) -> None:
    from torchgeo_bench import config as config_module

    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr('torchgeo_bench.main.main', capture)
    original_compose = config_module.compose_config

    def compose_with_optional_sections(
        overrides: Sequence[str] = (),
        *,
        config_name: str = 'config',
        default_model: str | None = 'rcf',
    ) -> DictConfig:
        legacy = original_compose(
            overrides, config_name=config_name, default_model=default_model
        )
        with open_dict(legacy.model):
            legacy.model.eval = {}
        with open_dict(legacy.eval.segmentation):
            legacy.eval.segmentation.layers = ['layer1']
        return legacy

    monkeypatch.setattr(config_module, 'compose_config', compose_with_optional_sections)
    config = validate_run_config(
        {
            'model': {'name': 'rcf'},
            'datasets': ['m-eurosat'],
            'runtime': {'device': 'cpu'},
            'segmentation': {'layers': ['layer1']},
        }
    )
    from torchgeo_bench.legacy_run import run

    run(config)
    assert received[0].model.eval == {}
    assert received[0].eval.segmentation.layers == ['layer1']


def test_legacy_adapter_preserves_preset_layers_when_schema_omits_them(
    monkeypatch: MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr('torchgeo_bench.main.main', capture)
    config = validate_run_config(
        {
            'model': {'name': 'torchgeo/resnet50_s2rgb_satlas_si'},
            'datasets': ['burn_scars'],
            'runtime': {'device': 'cpu'},
        }
    )
    from torchgeo_bench.legacy_run import run

    run(config)
    assert received[0].model.eval.segmentation.layers == [
        'layer4',
        'layer3',
        'layer2',
        'layer1',
    ]


def test_legacy_adapter_explicit_empty_layers_clear_preset(
    monkeypatch: MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr('torchgeo_bench.main.main', capture)
    config = validate_run_config(
        {
            'model': {'name': 'torchgeo/resnet50_s2rgb_satlas_si'},
            'datasets': ['burn_scars'],
            'runtime': {'device': 'cpu'},
            'segmentation': {'layers': []},
        }
    )
    from torchgeo_bench.legacy_run import run

    run(config)
    assert received[0].eval.segmentation.layers == []


def test_linear_only_is_rejected_by_legacy_adapter() -> None:
    with pytest.raises(SystemExit, match='2'):
        main(['run', '--model', 'rcf', '--dataset', 'm-eurosat', '--methods', 'linear'])


def test_unknown_config_field_fails_before_execution(tmp_path: Path) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        'model: {name: rcf}\ndatasets: [m-eurosat]\nrunntim: {}\n', encoding='utf-8'
    )
    with pytest.raises(SystemExit, match='2'):
        main(['run', '--config', str(path), '--dry-run'])


def test_runtime_failure_propagates_from_image_cli(monkeypatch: MonkeyPatch) -> None:
    def fail(_: object) -> None:
        raise FileNotFoundError('dataset missing')

    monkeypatch.setattr('torchgeo_bench.legacy_run.run', fail)
    with pytest.raises(FileNotFoundError, match='dataset missing'):
        main(['run', '--model', 'rcf', '--dataset', 'm-eurosat'])


def test_nested_linear_override_preserves_sibling_values(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / 'config.yaml'
    path.write_text(
        'model: {name: rcf}\ndatasets: [m-eurosat]\n'
        'classification:\n  knn_k: 7\n  linear:\n    refit_train_val: true\n',
        encoding='utf-8',
    )
    main(['run', '--config', str(path), '--no-refit-train-val', '--dry-run'])
    output = capsys.readouterr().out
    assert 'knn_k: 7' in output
    assert 'refit_train_val: false' in output


def test_catalogs_are_lightweight() -> None:
    assert 'timm/resnet50' in _model_names()
    assert len(_model_names()) > 1


def test_catalog_name_selection_and_unimplemented_commands(
    capsys: CaptureFixture[str],
) -> None:
    main(['models'])
    assert 'timm/resnet50' in capsys.readouterr().out
    main(['datasets'])
    assert 'm-eurosat' in capsys.readouterr().out
    main(['models', 'timm/resnet50'])
    assert '_target_:' in capsys.readouterr().out
    main(['datasets', 'm-eurosat'])
    dataset_detail = capsys.readouterr().out
    assert 'name: m-eurosat' in dataset_detail
    assert 'task: classification' in dataset_detail
    with pytest.raises(SystemExit, match='2'):
        main(['profile'])
    with pytest.raises(SystemExit, match='unknown model'):
        main(['models', 'unknown'])
    with pytest.raises(SystemExit, match='unknown dataset'):
        main(['datasets', 'unknown'])


def test_unknown_catalog_entries_fail_before_execution() -> None:
    with pytest.raises(SystemExit, match='2'):
        main(['run', '--model', 'unknown', '--dataset', 'm-eurosat'])
    with pytest.raises(SystemExit, match='2'):
        main(['run', '--model', 'rcf', '--dataset', 'unknown'])


def test_help_and_catalog_subprocesses_do_not_import_ml() -> None:
    code = (
        'import sys; from torchgeo_bench.image_cli import main; '
        'main(sys.argv[1:]); '
        "print([n for n in ('torch','torchgeo','pandas','numpy') if n in sys.modules])"
    )
    for args in (['models'], ['datasets']):
        result = subprocess.run(
            [sys.executable, '-c', code, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.rstrip().endswith('[]')


def test_config_help_is_available_without_selection(
    capsys: CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(['run', '--config-help'])
    assert error.value.code == 0
    assert 'title: RunConfig' in capsys.readouterr().out


def test_main_rejects_unknown_parser_command(monkeypatch: MonkeyPatch) -> None:
    import argparse

    monkeypatch.setattr(
        'torchgeo_bench.image_cli._parser',
        lambda: argparse.Namespace(
            parse_args=lambda _: argparse.Namespace(command='other')
        ),
    )
    with pytest.raises(SystemExit, match='not implemented'):
        main([])


def test_module_entrypoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, 'argv', ['torchgeo-bench', 'models', 'rcf'])
    runpy.run_module('torchgeo_bench.image_cli', run_name='__main__')
