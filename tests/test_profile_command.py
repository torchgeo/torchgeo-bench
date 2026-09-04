# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the standalone profile command."""

import argparse
import json
from collections.abc import Iterator

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from torchgeo_bench.commands import _profile_runtime
from torchgeo_bench.main import resolve_model_config


class _Band:
    def __init__(self, name: str) -> None:
        self.name = name


class _Dataset:
    rgb_bands = (_Band('red'), _Band('green'), _Band('blue'))

    def select_band_specs(self, bands: tuple[_Band, ...] | None) -> tuple[_Band, ...]:
        return self.rgb_bands if bands is None else tuple(bands)


class _Loader:
    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield {'image': torch.ones(4, 3, 8, 8)}


@pytest.mark.parametrize(
    (
        'model_data',
        'image_size',
        'interpolation',
        'expected_size',
        'expected_interpolation',
    ),
    [
        (
            {'dataset_overrides': {'toy': {'image_size': 96}}},
            None,
            None,
            96,
            'bilinear',
        ),
        ({}, None, None, 224, 'bilinear'),
        (
            {'dataset_overrides': {'toy': {'image_size': 96}}},
            128,
            'nearest',
            128,
            'nearest',
        ),
    ],
)
def test_input_settings_precedence(
    model_data: dict[str, object],
    image_size: int | None,
    interpolation: str | None,
    expected_size: int,
    expected_interpolation: str,
) -> None:
    cfg = OmegaConf.create(
        {'dataset': {'image_size': 224, 'interpolation': 'bilinear'}}
    )
    model_cfg = resolve_model_config(OmegaConf.create(model_data), 'toy')
    args = argparse.Namespace(
        image_size=image_size, interpolation=interpolation, normalization=None
    )

    effective = _profile_runtime._resolve_input_settings(args, cfg, model_cfg)

    assert effective[:2] == (expected_size, expected_interpolation)


def test_input_settings_preserve_model_normalization() -> None:
    cfg = OmegaConf.create(
        {'dataset': {'image_size': 224, 'interpolation': 'bilinear'}}
    )
    model_cfg = resolve_model_config(
        OmegaConf.create({'input_normalization': 'imagenet'}), 'toy'
    )
    args = argparse.Namespace(image_size=None, interpolation=None, normalization=None)

    effective = _profile_runtime._resolve_input_settings(args, cfg, model_cfg)

    assert effective[2:] == ('bandspec_zscore', 'imagenet')


def test_profile_emits_real_batch_metadata(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_profile_runtime, 'get_bench_dataset_class', lambda _: _Dataset)
    monkeypatch.setattr(_profile_runtime, 'list_model_configs', lambda: ['toy'])
    monkeypatch.setattr(_profile_runtime, 'list_datasets', lambda: ['toy-dataset'])
    monkeypatch.setattr(
        _profile_runtime, 'get_datasets', lambda **_: (None, _Loader(), None, None)
    )
    monkeypatch.setattr(
        _profile_runtime, 'compose_config', lambda _: OmegaConf.create({'model': {}})
    )
    monkeypatch.setattr(
        _profile_runtime,
        'resolve_model_config',
        lambda model, dataset: OmegaConf.create({}),
    )
    monkeypatch.setattr(
        _profile_runtime, 'instantiate', lambda *_, **__: nn.Conv2d(3, 3, 1)
    )
    args = argparse.Namespace(
        model='toy',
        dataset='toy-dataset',
        partition='default',
        device='cpu',
        bands='rgb',
        image_size=8,
        interpolation='bilinear',
        batch_size=4,
        warmup=0,
        measurements=1,
        precision='float32',
        count_flops=False,
        seed=0,
        normalization='bandspec_zscore',
    )

    _profile_runtime.run(args)
    record = json.loads(capsys.readouterr().out)
    assert record['model'] == 'toy'
    assert record['bands'] == ['red', 'green', 'blue']
    assert record['input_shape'] == [4, 3, 8, 8]
    assert record['profile']['batch_size'] == 4
    assert record['profile']['flops']['status'] == 'disabled'
    assert record['scope'].startswith('encoder inference')
