"""Tests for explicit construction of migrated model families."""

import pytest
import torch

from torchgeo_bench.models.build import (
    RCFModelConfig,
    TimmModelConfig,
    build_model,
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


def test_builder_rejects_settings_for_other_model_family() -> None:
    with pytest.raises(ValueError, match="rcf settings"):
        build_model('timm/resnet18', _bands(3), rcf=RCFModelConfig())


def test_builder_rejects_unmigrated_model() -> None:
    with pytest.raises(ValueError, match='not migrated'):
        build_model('torchgeo/resnet18', _bands(3))


def test_model_config_rejects_unknown_fields() -> None:
    with pytest.raises(TypeError, match='unexpected'):
        TimmModelConfig(model_name='resnet18', unexpected=True)  # type: ignore[call-arg]
