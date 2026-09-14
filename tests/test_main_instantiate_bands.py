"""Unit test for the typed model construction boundary."""

from unittest import mock

import torch

from tests.support.runner import _DictTensorDataset, _synthetic_loaders
from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.main import instantiate_dataset_model
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.presets import resolve_run_config


def test_empirical_rcf_receives_run_seed_and_actual_dataset(monkeypatch) -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"mode": "empirical"}},
            "datasets": ["m-eurosat"],
            "runtime": {"seed": 17, "device": "cpu"},
            "input": {"normalization": "none"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset, *_ = _synthetic_loaders()
    captured = []

    def build(settings, bands, *, normalization):
        captured.append((settings, bands, normalization))
        return mock.Mock()

    monkeypatch.setattr("torchgeo_bench.models.rcf.RCFModelSettings.build", build)
    instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )
    settings, bands, normalization = captured[0]
    assert settings.dataset is dataset
    assert settings.seed == 17
    assert all(isinstance(band, BandSpec) for band in bands)
    assert normalization == "identity"


class StrictCustomModel(BenchModel):
    def __init__(self, bands, normalization):
        super().__init__(bands=bands, normalization=normalization)

    def _forward_patch_features(self, images):
        return images.mean(dim=(-2, -1))


def test_custom_constructor_never_receives_model_metadata() -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "custom", "target": f"{__name__}.StrictCustomModel"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            "segmentation": {"layers": [], "learning_rate": 0.01},
            "input": {
                "image_size": None,
                "interpolation": "nearest",
                "normalization": "minmax_zscore",
            },
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset, *_ = _synthetic_loaders()
    model = instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )
    assert isinstance(model, StrictCustomModel)
    assert len(model.bands) == 3


def test_temporal_input_uses_channel_dimension_not_time_dimension() -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"features": 8}},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset = _DictTensorDataset(torch.rand(4, 2, 3, 8, 8), torch.zeros(4))
    model = instantiate_dataset_model(
        config, preset, get_bench_dataset_class("m-eurosat")(), dataset, torch.device("cpu")
    )
    assert model.num_channels == 3
