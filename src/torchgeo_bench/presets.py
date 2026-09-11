"""Typed model presets and effective image settings.

Only explicitly supplied values override preset defaults. Runtime objects such as
BandSpecs and empirical-RCF datasets are passed directly to construction, never YAML.
"""

import importlib
from copy import deepcopy
from typing import Any, Literal

from pydantic import Field

from .config_schema import (
    ClassificationConfig,
    InputConfig,
    ModelConfig,
    RunConfig,
    SegmentationConfig,
    StrictModel,
    load_yaml,
)

NORMALIZATIONS = {
    "dataset": "bandspec_zscore",
    "model": "model_native",
    "minmax": "minmax",
    "minmax_zscore": "minmax_zscore",
    "none": "identity",
}


def merge_settings(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit YAML settings; nulls and empty lists replace prior values."""
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_settings(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


class PresetDefaults(StrictModel):
    """Constructor and benchmark defaults for one dataset or model."""

    kwargs: dict[str, Any] = Field(default_factory=dict)
    input: InputConfig = Field(default_factory=InputConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)


class ModelPreset(PresetDefaults):
    """A named constructor, separate from preprocessing and evaluation metadata."""

    name: str
    target: str
    track: Literal["image", "coord"] = "image"
    seed_from_run: bool = False
    dataset_overrides: dict[str, PresetDefaults] = Field(default_factory=dict)

    def for_dataset(self, dataset: str) -> "ModelPreset":
        """Apply a dataset override without mutating the shared preset."""
        values = self.model_dump(exclude_unset=True)
        values.pop("dataset_overrides", None)
        override = self.dataset_overrides.get(dataset)
        if override is not None:
            values = merge_settings(values, override.model_dump(exclude_unset=True))
        return ModelPreset.model_validate(values)


def _legacy_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate the finite packaged preset vocabulary during staged migration."""
    values = deepcopy(raw)
    result: dict[str, Any] = {}
    inputs = {key: values.pop(key) for key in ("image_size", "interpolation") if key in values}
    if inputs:
        result["input"] = inputs
    evaluation = values.pop("eval", {})
    if "c_range" in evaluation:
        start, stop, count = evaluation.pop("c_range")
        result["classification"] = {
            "linear": {"c_log10_start": start, "c_log10_stop": stop, "c_count": count}
        }
    if "segmentation" in evaluation:
        segmentation = evaluation.pop("segmentation")
        for old, new in (
            ("head_type", "head"),
            ("lr", "learning_rate"),
            ("lr_scheduler", "scheduler"),
        ):
            if old in segmentation:
                segmentation[new] = segmentation.pop(old)
        result["segmentation"] = segmentation
    if evaluation:
        raise ValueError(f"Unsupported preset evaluation defaults: {sorted(evaluation)}")
    result["kwargs"] = values
    return result


def load_model_preset(selection: ModelConfig, *, seed: int = 0) -> ModelPreset:
    """Load a packaged preset or a custom constructor without importing weights."""
    if selection.target is not None:
        return ModelPreset(name=selection.name, target=selection.target, kwargs=selection.kwargs)
    from .config import model_config_path

    raw = load_yaml(model_config_path(selection.name))
    if "target" in raw:
        preset = ModelPreset.model_validate(raw)
    else:
        name, target = raw.pop("name"), raw.pop("_target_")
        overrides = raw.pop("dataset_overrides", {})
        seed_from_run = raw.get("seed") == "${seed}"
        if seed_from_run:
            raw.pop("seed")
        preset = ModelPreset(
            name=name,
            target=target,
            track="coord" if ".coordbench." in target else "image",
            seed_from_run=seed_from_run,
            dataset_overrides={
                key: PresetDefaults.model_validate(_legacy_defaults(value))
                for key, value in overrides.items()
            },
            **_legacy_defaults(raw),
        )
    kwargs = dict(preset.kwargs)
    if preset.seed_from_run:
        kwargs["seed"] = seed
    kwargs.update(selection.kwargs)
    return preset.model_copy(update={"kwargs": kwargs})


def resolve_run_config(config: RunConfig, dataset: str) -> tuple[RunConfig, ModelPreset]:
    """Resolve preset and dataset defaults beneath explicitly supplied settings."""
    preset = load_model_preset(config.model, seed=config.runtime.seed).for_dataset(dataset)
    preset = preset.model_copy(update={"kwargs": {**preset.kwargs, **config.model.kwargs}})
    if preset.track != "image":
        raise ValueError(f"{config.model.name!r} is a coordinate encoder; use 'coord'")
    defaults = {
        key: getattr(preset, key).model_dump(exclude_unset=True)
        for key in ("input", "classification", "segmentation")
    }
    effective = RunConfig.model_validate(merge_settings(defaults, config.model_dump_yaml()))
    return effective, preset


def build_model(preset: ModelPreset, **runtime_options: Any) -> Any:
    """Construct one model; nested target-like kwargs remain ordinary mappings."""
    options = {**preset.kwargs, **runtime_options}
    if preset.target in {
        "torchgeo_bench.models.TimmPatchBenchModel",
        "torchgeo_bench.models.RCFBench",
    }:
        from .models.build import (
            RCFModelConfig,
            TimmModelConfig,
            build_rcf_model,
            build_timm_model,
        )

        bands = options.pop("bands")
        normalization = options.pop("normalization", "bandspec_zscore")
        if preset.target.endswith("TimmPatchBenchModel"):
            options.pop("seed", None)
            return build_timm_model(TimmModelConfig(**options), bands, normalization=normalization)
        return build_rcf_model(RCFModelConfig(**options), bands, normalization=normalization)
    module, _, symbol = preset.target.rpartition(".")
    constructor = getattr(importlib.import_module(module), symbol)
    return constructor(**options)
