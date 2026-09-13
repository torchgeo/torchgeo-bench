"""Offline coverage of the typed preset boundary."""

import subprocess
import sys

import pytest

from torchgeo_bench.config import list_model_configs
from torchgeo_bench.config_schema import ModelConfig, RunConfig
from torchgeo_bench.presets import ModelPreset, load_model_preset, resolve_run_config


@pytest.mark.parametrize("name", list_model_configs())
def test_packaged_preset_is_typed_without_loading_weights(name: str) -> None:
    preset = load_model_preset(ModelConfig(name=name), seed=17)
    assert isinstance(preset, ModelPreset)
    assert preset.target
    assert "name" not in preset.kwargs
    assert "eval" not in preset.kwargs
    assert "dataset_overrides" not in preset.kwargs
    assert "image_size" not in preset.kwargs
    assert "interpolation" not in preset.kwargs
    assert "${" not in str(preset.model_dump())


def test_rcf_seed_is_explicit_and_overridable() -> None:
    assert load_model_preset(ModelConfig(name="rcf"), seed=17).kwargs["seed"] == 17
    assert (
        load_model_preset(ModelConfig(name="rcf", kwargs={"seed": 9}), seed=17).kwargs["seed"] == 9
    )


def test_preset_layers_do_not_override_an_explicit_empty_selection() -> None:
    omitted = RunConfig(model=ModelConfig(name="timm/resnet50"), datasets=["caffe"])
    explicit = RunConfig.model_validate(
        {
            **omitted.model_dump_yaml(),
            "segmentation": {"layers": [], "cache_features": False},
            "input": {"image_size": None},
        }
    )
    effective, _ = resolve_run_config(omitted, "caffe")
    assert effective.segmentation.layers == ["layer4", "layer3", "layer2", "layer1"]
    effective, _ = resolve_run_config(explicit, "caffe")
    assert effective.segmentation.layers == []
    assert not effective.segmentation.cache_features
    assert effective.input.image_size is None
    assert explicit.model_dump_yaml()["input"] == {"image_size": None}


def test_custom_model_keeps_constructor_options_separate() -> None:
    selection = ModelConfig(
        name="custom",
        target="custom_model.Model",
        kwargs={"options": {"_target_": "ordinary.mapping"}},
    )
    preset = load_model_preset(selection)
    assert preset.kwargs == {"options": {"_target_": "ordinary.mapping"}}
    assert preset.target == "custom_model.Model"


def test_configuration_import_and_resolution_do_not_import_omegaconf() -> None:
    code = """
import sys
from torchgeo_bench.config import list_model_configs
from torchgeo_bench.config_schema import ModelConfig
from torchgeo_bench.presets import load_model_preset
for name in list_model_configs():
    load_model_preset(ModelConfig(name=name))
assert not set(('omegaconf', 'torch', 'torchgeo', 'numpy', 'pandas')) & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True)
