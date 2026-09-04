# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Implementation of the image benchmark command."""

import argparse
from typing import Any

import yaml

from torchgeo_bench.config_schema import RunConfig, load_yaml, validate_run_config


def _set(overrides: dict[str, Any], section: str, key: str, value: Any) -> None:
    """Set one explicit CLI override in a nested mapping."""
    if '.' in section:
        parent, child = section.split('.', maxsplit=1)
        overrides.setdefault(parent, {}).setdefault(child, {})[key] = value
    else:
        overrides.setdefault(section, {})[key] = value


def _run_mapping(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901 - flat flag table
    """Translate supplied run flags into the schema's mapping."""
    overrides: dict[str, Any] = {}
    if getattr(args, 'model', None) is not None:
        overrides['model'] = {'name': args.model}
    if getattr(args, 'datasets', None) is not None:
        overrides['datasets'] = args.datasets
    for name in ('device', 'batch_size', 'workers', 'seed', 'verbose'):
        if hasattr(args, name):
            _set(overrides, 'runtime', name, getattr(args, name))
    for name in ('bands', 'image_size', 'normalization', 'partition', 'time_steps'):
        if hasattr(args, name):
            _set(overrides, 'input', name, getattr(args, name))
    for name in ('methods', 'knn_k', 'knn_device', 'bootstrap_samples'):
        if hasattr(args, name):
            _set(overrides, 'classification', name, getattr(args, name))
    if hasattr(args, 'refit_train_val'):
        _set(
            overrides, 'classification.linear', 'refit_train_val', args.refit_train_val
        )
    if hasattr(args, 'temp_scale'):
        _set(overrides, 'classification.calibration', 'temp_scale', args.temp_scale)
    if hasattr(args, 'resume'):
        _set(overrides, 'output', 'resume', args.resume)
    return overrides


def _apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply known CLI fields with explicit section handling."""
    result = dict(base)
    for key, value in overrides.items():
        if key in {'model', 'input', 'classification', 'runtime', 'output'}:
            section = dict(result.get(key, {}))
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict) and isinstance(
                    section.get(nested_key), dict
                ):
                    merged = dict(section[nested_key])
                    merged.update(nested_value)
                    section[nested_key] = merged
                else:
                    section[nested_key] = nested_value
            result[key] = section
        else:
            result[key] = value
    return result


def _load_run(
    args: argparse.Namespace, model_names: list[str], datasets: tuple[str, ...]
) -> RunConfig:
    """Load, validate, and return the selected image configuration."""
    config_path = getattr(args, 'config', None)
    base = load_yaml(config_path) if config_path else {}
    values = _apply_overrides(base, _run_mapping(args))
    if getattr(args, 'config_help', False):
        print(yaml.safe_dump(RunConfig.model_json_schema(), sort_keys=False), end='')
        raise SystemExit(0)
    if 'model' not in values or 'name' not in values['model']:
        raise ValueError(
            'model is required; pass --model or set model.name in --config'
        )
    if 'datasets' not in values:
        raise ValueError(
            'at least one dataset is required; pass --dataset or set datasets'
        )
    unknown_models = (
        [] if values['model']['name'] in model_names else [values['model']['name']]
    )
    unknown_datasets = [name for name in values['datasets'] if name not in datasets]
    if unknown_models or unknown_datasets:
        raise ValueError(
            f'unknown model or dataset: models={unknown_models}, datasets={unknown_datasets}'
        )
    return validate_run_config(values)


def run(
    args: argparse.Namespace, model_names: list[str], datasets: tuple[str, ...]
) -> None:
    """Validate and execute one image benchmark."""
    config = _load_run(args, model_names, datasets)
    if getattr(args, 'dry_run', False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end='')
        return
    methods = set(config.classification.methods)
    if methods != {'knn', 'linear'} and methods != {'knn'}:
        raise ValueError(
            'legacy runner currently supports methods [knn, linear] or [knn]'
        )
    from torchgeo_bench import legacy_run

    legacy_run.run(config)
