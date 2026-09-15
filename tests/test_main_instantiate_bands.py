"""Unit test for the typed model construction boundary."""

from unittest import mock

import pytest
import torch

from tests.support.runner import _DictTensorDataset, _synthetic_splits, make_loaded_split
from torchgeo_bench.config.presets import resolve_run_config
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.main import instantiate_dataset_model
from torchgeo_bench.models.interface import BenchModel


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
    train, *_ = _synthetic_splits()
    captured = []

    def build(settings, bands, *, normalization):
        captured.append((settings, bands, normalization))
        return mock.Mock()

    monkeypatch.setattr("torchgeo_bench.models.rcf.RCFModelSettings.build", build)
    instantiate_dataset_model(config, preset, train, torch.device("cpu"))
    settings, bands, normalization = captured[0]
    assert settings.dataset is train.dataset
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
    train, *_ = _synthetic_splits()
    model = instantiate_dataset_model(config, preset, train, torch.device("cpu"))
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
    train = make_loaded_split(dataset, "pastis", time_steps=2)
    model = instantiate_dataset_model(config, preset, train, torch.device("cpu"))
    assert model.num_channels == 3


def test_band_spec_count_must_match_tensor_channels() -> None:
    config = RunConfig.model_validate(
        {"model": {"name": "rcf"}, "datasets": ["m-eurosat"], "runtime": {"device": "cpu"}}
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset = _DictTensorDataset(torch.zeros(1, 4, 8, 8), torch.zeros(1))
    with (
        mock.patch("torchgeo_bench.main.build_model") as build,
        pytest.raises(ValueError, match="BandSpec count 3 != tensor channel count 4"),
    ):
        instantiate_dataset_model(config, preset, make_loaded_split(dataset), torch.device("cpu"))
    build.assert_not_called()


def test_loaded_band_objects_reach_model_without_another_selection() -> None:
    config = RunConfig.model_validate(
        {
            "model": {"name": "custom", "target": f"{__name__}.StrictCustomModel"},
            "datasets": ["m-eurosat"],
            "input": {"bands": "all"},
            "runtime": {"device": "cpu"},
        }
    )
    config, preset = resolve_run_config(config, "m-eurosat")
    dataset = _DictTensorDataset(torch.zeros(2, 2, 8, 8), torch.zeros(2))
    train = make_loaded_split(dataset, bands=("nir", "red"))
    bench_cls = get_bench_dataset_class("m-eurosat")
    with (
        mock.patch.object(
            bench_cls, "resolve_band_specs", side_effect=AssertionError("reselected")
        ),
        mock.patch.object(bench_cls, "select_band_specs", side_effect=AssertionError("reselected")),
    ):
        model = instantiate_dataset_model(config, preset, train, torch.device("cpu"))
    assert [band.name for band in model.bands] == ["nir", "red"]
    assert all(
        actual is expected for actual, expected in zip(model.bands, train.bands, strict=True)
    )


@pytest.mark.parametrize("shape", [(2, 3, 8, 8), (3, 4, 8, 8)])
def test_model_rejects_layout_that_disagrees_with_loaded_metadata(shape: tuple[int, ...]) -> None:
    config = RunConfig.model_validate({"model": {"name": "rcf"}, "datasets": ["pastis"]})
    config, preset = resolve_run_config(config, "pastis")
    train = make_loaded_split(
        _DictTensorDataset(torch.ones(1, *shape), torch.zeros(1)),
        "pastis",
        time_steps=3,
    )
    with pytest.raises(ValueError, match=r"expected TCHW|BandSpec count"):
        instantiate_dataset_model(config, preset, train, torch.device("cpu"))
