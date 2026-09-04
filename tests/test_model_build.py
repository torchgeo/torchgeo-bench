# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for explicit construction of migrated model families."""

from typing import Any, cast

import pytest
import torch

from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.models.build import (
    RCFModelConfig,
    TimmModelConfig,
    build_rcf_model,
    build_timm_model,
)
from torchgeo_bench.models.rcf import RCFBench
from torchgeo_bench.models.timm import TimmPatchBenchModel

from .test_bench_model import _bands


def test_timm_builder_matches_direct_constructor() -> None:
    bands = _bands(3)
    config = TimmModelConfig(model_name='resnet18', pretrained=False)
    torch.manual_seed(7)
    expected = TimmPatchBenchModel(
        bands=bands,
        model_name='resnet18',
        pretrained=False,
        global_pool='avg',
        normalization='identity',
    )
    torch.manual_seed(7)
    actual = build_timm_model(config, bands, normalization='identity')

    assert isinstance(actual, TimmPatchBenchModel)
    assert all(
        torch.equal(expected.state_dict()[key], value)
        for key, value in actual.state_dict().items()
    )
    inputs = torch.rand(2, 3, 32, 32)
    with torch.inference_mode():
        assert torch.equal(expected(inputs), actual(inputs))


def test_timm_transformer_builder_constructs_without_checkpoint() -> None:
    config = TimmModelConfig(
        model_name='vit_tiny_patch16_224',
        pretrained=False,
        auto_resize=True,
        target_size=224,
    )
    bands = _bands(3)
    torch.manual_seed(13)
    expected = TimmPatchBenchModel(
        bands=bands,
        model_name='vit_tiny_patch16_224',
        pretrained=False,
        auto_resize=True,
        target_size=224,
        normalization='identity',
    ).eval()
    torch.manual_seed(13)
    model = build_timm_model(config, bands, normalization='identity').eval()

    with torch.inference_mode():
        inputs = torch.rand(2, 3, 32, 32)
        features = model(inputs)
        expected_features = expected(inputs)

    assert features.shape[0] == 2
    assert features.ndim == 2
    assert torch.equal(features, expected_features)


def test_instantiate_routes_timm_preset_to_explicit_builder() -> None:
    config = compose_config(['model=timm/resnet18', 'model.pretrained=false'])
    model = instantiate(config.model, bands=_bands(3), normalization='identity')

    assert isinstance(model, TimmPatchBenchModel)
    assert model.pretrained is False


def test_instantiate_routes_rcf_preset_to_explicit_builder() -> None:
    config = compose_config(['model=rcf'])
    model = instantiate(config.model, bands=_bands(3), normalization='identity')

    assert isinstance(model, RCFBench)
    assert model.rcf.weights.shape[0] == 256


def test_rcf_builder_matches_direct_constructor() -> None:
    bands = _bands(3)
    config = RCFModelConfig(features=16, kernel_size=3, seed=7)
    torch.manual_seed(11)
    expected = RCFBench(
        bands=bands, features=16, kernel_size=3, seed=7, normalization='identity'
    )
    torch.manual_seed(11)
    actual = build_rcf_model(config, bands, normalization='identity')

    assert isinstance(actual, RCFBench)
    assert torch.equal(expected.rcf.weights, actual.rcf.weights)
    assert torch.equal(expected.rcf.biases, actual.rcf.biases)


def test_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match='unexpected'):
        cast(Any, TimmModelConfig)(model_name='resnet18', unexpected=True)


@pytest.mark.parametrize(
    'kwargs',
    [
        {'model_name': ''},
        {'model_name': 'resnet18', 'target_size': 0},
        {'model_name': 'resnet18', 'global_pool': 'invalid'},
        {'model_name': 'resnet18', 'input_normalization': 'invalid'},
        {'model_name': 'resnet18', 'pretrained': 1},
        {'model_name': 'resnet18', 'auto_resize': 1},
        {'model_name': 'resnet18', 'target_size': '224'},
    ],
)
def test_timm_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        cast(Any, TimmModelConfig)(**kwargs)


def test_rcf_empirical_config_requires_dataset() -> None:
    with pytest.raises(ValueError, match='dataset must be provided'):
        RCFModelConfig(mode='empirical')


@pytest.mark.parametrize('kwargs', [{'features': '512'}, {'kernel_size': '3'}])
def test_rcf_config_rejects_non_integer_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError, match='must be integers'):
        cast(Any, RCFModelConfig)(**kwargs)


def test_legacy_model_translation_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match='Unknown settings'):
        instantiate(
            {'_target_': 'torchgeo_bench.models.RCFBench', 'unknown': True},
            bands=_bands(3),
        )


@pytest.mark.parametrize(
    'kwargs',
    [
        {'features': 3},
        {'kernel_size': 0},
        {'mode': 'invalid'},
        {'stats_mode': 'invalid'},
    ],
)
def test_rcf_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match='must be|mode must|stats_mode'):
        cast(Any, RCFModelConfig)(**kwargs)
