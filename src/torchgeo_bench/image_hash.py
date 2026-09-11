"""Image resume serialization compatible with pre-migration result files.

Version 1 hashed the model recipe *and* the user evaluation section, including
obsolete visualization defaults. Preserve that representation, not a Pydantic
dump. Legacy composition used integer C endpoints; the image adapter used floats
and copied preset layers. Both are accepted only when their effective settings
match the current run. New rows retain the public image CLI's fingerprint when
its effective recipe is unchanged. Additive profile/ID passes and dataset
selection stay out.
"""

import hashlib
import json
from copy import deepcopy
from itertools import product
from typing import Any, Literal

from .config_schema import RunConfig, SegmentationConfig
from .presets import (
    NORMALIZATIONS,
    PresetDefaults,
    load_model_preset,
    merge_settings,
    resolve_run_config,
)

_SEGMENTATION_KEYS = {
    "head": "head_type",
    "learning_rate": "lr",
    "scheduler": "lr_scheduler",
}
_MODEL_METADATA = {"_target_", "name", "eval", "image_size", "interpolation", "dataset_overrides"}


def _segmentation_payload(config: SegmentationConfig) -> dict[str, Any]:
    values = {
        _SEGMENTATION_KEYS.get(key, key): value
        for key, value in config.model_dump().items()
        if key != "ignore_index"
    }
    values["criterion"] = {
        "_target_": "torch.nn.CrossEntropyLoss",
        "ignore_index": config.ignore_index,
    }
    values.update(save_viz=False, viz_dir="viz", n_viz_samples=8)
    return values


def _preset_payload(defaults: PresetDefaults) -> dict[str, Any]:
    """Retain the old CSV fingerprint representation, not its configuration engine."""
    values = {**defaults.kwargs, **defaults.input.model_dump(exclude_unset=True)}
    evaluation: dict[str, Any] = {}
    classification = defaults.classification
    if "linear" in classification.model_fields_set:
        linear = classification.linear
        evaluation["c_range"] = [
            int(value) if float(value).is_integer() else value
            for value in (linear.c_log10_start, linear.c_log10_stop)
        ] + [linear.c_count]
    for key in classification.model_fields_set - {"linear", "methods"}:
        legacy_key = "bootstrap" if key == "bootstrap_samples" else key
        evaluation[legacy_key] = classification.model_dump()[key]
    segmentation = defaults.segmentation.model_dump(exclude_unset=True)
    if segmentation:
        evaluation["segmentation"] = {
            _SEGMENTATION_KEYS.get(key, key): value
            for key, value in segmentation.items()
            if key != "ignore_index"
        }
        if "ignore_index" in segmentation:
            evaluation["segmentation"]["criterion"] = {
                "_target_": "torch.nn.CrossEntropyLoss",
                "ignore_index": defaults.segmentation.ignore_index,
            }
    if evaluation:
        values["eval"] = evaluation
    return values


def _model_payload(config: RunConfig) -> dict[str, Any]:
    if config.model.target is not None:
        return {"_target_": config.model.target, "name": config.model.name, **config.model.kwargs}
    preset = load_model_preset(config.model, seed=config.runtime.seed)
    values = {"_target_": preset.target, "name": preset.name, **_preset_payload(preset)}
    if preset.dataset_overrides:
        values["dataset_overrides"] = {
            name: _preset_payload(override) for name, override in preset.dataset_overrides.items()
        }
    return values


def _historical_payload(config: RunConfig, style: Literal["legacy", "image"]) -> dict[str, Any]:
    """Serialize the two historical CLI formats, retaining numeric JSON types."""
    model = _model_payload(config)
    inputs = config.input.model_dump()
    inputs["normalization"] = NORMALIZATIONS[config.input.normalization]
    inputs.update(batch_size=config.runtime.batch_size, num_workers=config.runtime.workers)
    classification = config.classification
    linear = classification.linear
    start, stop = linear.c_log10_start, linear.c_log10_stop
    c_range = [
        int(value) if style == "legacy" and value.is_integer() else value
        for value in (float(start), float(stop))
    ] + [linear.c_count]
    segmentation = _segmentation_payload(config.segmentation)
    if style == "image":
        if "layers" not in config.segmentation.model_fields_set:
            segmentation["layers"] = model.get("eval", {}).get("segmentation", {}).get("layers", [])
        for recipe in [model, *model.get("dataset_overrides", {}).values()]:
            if "image_size" in config.input.model_fields_set and "image_size" in recipe:
                recipe["image_size"] = config.input.image_size
            if "interpolation" in config.input.model_fields_set:
                recipe.pop("interpolation", None)
    return {
        "version": 1,
        "seed": config.runtime.seed,
        "device": config.runtime.device,
        "dataset": inputs,
        "eval": {
            "bootstrap": classification.bootstrap_samples,
            "c_range": c_range,
            "merge_val": linear.refit_train_val,
            "skip_linear": "linear" not in classification.methods,
            "knn_k": classification.knn_k,
            "knn_device": classification.knn_device,
            "calibration": classification.calibration.model_dump(),
            "segmentation": segmentation,
        },
        "model": model,
    }


def _legacy_canonical_payload(config: RunConfig) -> dict[str, Any]:
    """Retain the first typed runner's precedence-aware version-1 representation."""
    payload = _historical_payload(config, "legacy")
    if config.model.target is not None:
        # Custom kwargs are opaque constructor data, including names such as "eval".
        payload["model"] = config.model.model_dump()
        return payload
    model = payload["model"]
    evaluation = payload["eval"]
    # Old model defaults overrode user settings. Replacing only conflicting defaults
    # keeps equivalent keys while preventing a collision when precedence changes.
    for recipe in [model, *model.get("dataset_overrides", {}).values()]:
        for key in ("image_size", "interpolation"):
            if key in config.input.model_fields_set and key in recipe:
                recipe[key] = getattr(config.input, key)
        for key, value in config.model.kwargs.items():
            if key in recipe:
                recipe[key] = value
        _override_evaluation_defaults(recipe.get("eval", {}), config, evaluation)
    return payload


def _resume_config_payload(config: RunConfig) -> dict[str, Any]:
    """Retain the public CLI fingerprint unless effective preset behavior changed."""
    legacy = _legacy_canonical_payload(config)
    if config.model.target is not None:
        return legacy
    preset = load_model_preset(config.model, seed=config.runtime.seed)
    if preset.track != "image":
        return legacy
    image = _historical_payload(config, "image")
    # Consider every recipe, not the selected datasets: selection is not a hash input.
    recipes = ("", *preset.dataset_overrides)
    if all(
        _matches_effective(image, config, dataset, segmentation=segmentation, image=True)
        for dataset in recipes
        for segmentation in (False, True)
    ):
        return image
    return legacy


def _override_evaluation_defaults(
    defaults: dict[str, Any], config: RunConfig, evaluation: dict[str, Any]
) -> None:
    if "c_range" in defaults:
        for index, key in enumerate(("c_log10_start", "c_log10_stop", "c_count")):
            if key in config.classification.linear.model_fields_set:
                defaults["c_range"][index] = evaluation["c_range"][index]
    for key in config.segmentation.model_fields_set:
        old_key = "criterion" if key == "ignore_index" else _SEGMENTATION_KEYS.get(key, key)
        seg_defaults = defaults.get("segmentation", {})
        if old_key in seg_defaults:
            seg_defaults[old_key] = evaluation["segmentation"][old_key]


def _hash_payload(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _resume_config_hash(config: RunConfig) -> str:
    """Fingerprint result-affecting image settings, excluding additive measurements."""
    return _hash_payload(_resume_config_payload(config))


def _matches_effective(
    payload: dict[str, Any], config: RunConfig, dataset: str, *, segmentation: bool, image: bool
) -> bool:
    """Reject an old key when adapter precedence would have computed different results."""
    effective, preset = resolve_run_config(config, dataset)
    model = payload["model"]
    resolved_model = merge_settings(model, model.get("dataset_overrides", {}).get(dataset, {}))
    kwargs = {key: value for key, value in resolved_model.items() if key not in _MODEL_METADATA}
    if kwargs != preset.kwargs or model["_target_"] != preset.target:
        return False
    inputs = payload["dataset"]
    if any(
        resolved_model.get(key, inputs[key]) != getattr(effective.input, key)
        for key in ("image_size", "interpolation")
    ):
        return False
    evaluation, model_eval = payload["eval"], model.get("eval", {})
    if segmentation:
        previous = merge_settings(
            model_eval.get("segmentation", {}) if image else evaluation["segmentation"],
            evaluation["segmentation"] if image else model_eval.get("segmentation", {}),
        )
        return previous == _segmentation_payload(effective.segmentation)
    linear = effective.classification.linear
    previous_range = (
        evaluation["c_range"] if image else model_eval.get("c_range", evaluation["c_range"])
    )
    previous_k = evaluation["knn_k"] if image else model_eval.get("knn_k", evaluation["knn_k"])
    return (
        previous_range == [linear.c_log10_start, linear.c_log10_stop, linear.c_count]
        and previous_k == effective.classification.knn_k
    )


def compatible_hashes(config: RunConfig, dataset: str, *, segmentation: bool) -> set[str]:
    """Return only historical keys whose effective recipe matches this dataset's run."""
    hashes = {_resume_config_hash(config), _hash_payload(_legacy_canonical_payload(config))}
    if config.model.target is not None:
        return hashes
    for style in ("legacy", "image"):
        payload = _historical_payload(config, style)
        if not _matches_effective(
            payload, config, dataset, segmentation=segmentation, image=style == "image"
        ):
            continue
        # JSON distinguishes 4 from 4.0 even though both specify the same log10 C.
        endpoints = payload["eval"]["c_range"][:2]
        choices = [
            [int(value), float(value)] if float(value).is_integer() else [value]
            for value in endpoints
        ]
        for start, stop in product(*choices):
            candidate = deepcopy(payload)
            candidate["eval"]["c_range"][:2] = [start, stop]
            hashes.add(_hash_payload(candidate))
    return hashes
