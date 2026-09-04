"""Tests for explicit construction of migrated model families."""

import pytest
import torch

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
    assert all(torch.equal(expected.state_dict()[key], value) for key, value in actual.state_dict().items())


def test_timm_transformer_builder_constructs_without_checkpoint() -> None:
    config = TimmModelConfig(
        model_name='vit_tiny_patch16_224', pretrained=False, auto_resize=True, target_size=224
    )
    model = build_timm_model(config, _bands(3), normalization='identity').eval()

    with torch.inference_mode():
        features = model(torch.rand(2, 3, 32, 32))

    assert features.shape[0] == 2
    assert features.ndim == 2


def test_rcf_builder_matches_direct_constructor() -> None:
    bands = _bands(3)
    config = RCFModelConfig(features=16, kernel_size=3, seed=7)
    torch.manual_seed(11)
    expected = RCFBench(
        bands=bands,
        features=16,
        kernel_size=3,
        seed=7,
        normalization='identity',
    )
    torch.manual_seed(11)
    actual = build_rcf_model(config, bands, normalization='identity')

    assert isinstance(actual, RCFBench)
    assert torch.equal(expected.rcf.weights, actual.rcf.weights)
    assert torch.equal(expected.rcf.biases, actual.rcf.biases)


def test_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match='unexpected'):
        TimmModelConfig(model_name='resnet18', unexpected=True)  # type: ignore[call-arg]


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
    with pytest.raises(ValueError):
        RCFModelConfig(**kwargs)  # type: ignore[arg-type]
