"""Configuration loading and ``_target_`` instantiation.

The benchmark's config system is one typed settings dataclass
(:class:`torchgeo_bench.settings.RunSettings` or ``FlopsSettings``), an
optional ``--config PATH`` YAML of uncommon settings, one model YAML
selected from ``conf/model/``, and a nested-mapping of explicit CLI-flag
overrides -- composed in that precedence order (flags win).

There is no ``key=value`` dotlist or ``+``/``++`` override syntax: CLI flags
are translated by :mod:`torchgeo_bench.cli` directly into a nested mapping
and merged with :func:`torchgeo_bench.settings.merge`, which also enforces
that a mapping only ever targets a real settings field.
"""

import importlib
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Any

from torchgeo_bench.settings import FlopsSettings, RunSettings, load_yaml, merge

CONF_DIR = Path(str(files("torchgeo_bench") / "conf"))


class ModelConfigError(ValueError):
    """Raised when a model YAML does not match the supported structure."""


def validate_model_config(config: Mapping[str, Any], *, source: str | Path) -> None:
    """Validate the shared structure around heterogeneous model kwargs."""
    target = config.get("_target_")
    if not isinstance(target, str) or not target or "." not in target:
        raise ModelConfigError(f"{source}: '_target_' must be a dotted import path.")

    name = config.get("name")
    if not isinstance(name, str) or not name:
        raise ModelConfigError(f"{source}: 'name' must be a non-empty string.")

    eval_config = config.get("eval")
    if eval_config is not None and not isinstance(eval_config, Mapping):
        raise ModelConfigError(f"{source}: 'eval' must be a mapping.")
    if isinstance(eval_config, Mapping):
        c_range = eval_config.get("c_range")
        if c_range is not None and (
            not isinstance(c_range, list)
            or len(c_range) != 3
            or not all(
                not isinstance(value, bool) and isinstance(value, int | float)
                for value in c_range
            )
        ):
            raise ModelConfigError(f"{source}: 'eval.c_range' must be a three-number list.")

        segmentation = eval_config.get("segmentation")
        if segmentation is not None and not isinstance(segmentation, Mapping):
            raise ModelConfigError(f"{source}: 'eval.segmentation' must be a mapping.")
        if isinstance(segmentation, Mapping):
            layers = segmentation.get("layers")
            if layers is not None and (
                not isinstance(layers, list)
                or not all(isinstance(layer, str) and layer for layer in layers)
            ):
                raise ModelConfigError(
                    f"{source}: 'eval.segmentation.layers' must be a list of strings."
                )

    dataset_overrides = config.get("dataset_overrides")
    if dataset_overrides is not None and (
        not isinstance(dataset_overrides, Mapping)
        or not all(
            isinstance(dataset, str) and isinstance(values, Mapping)
            for dataset, values in dataset_overrides.items()
        )
    ):
        raise ModelConfigError(
            f"{source}: 'dataset_overrides' must map dataset names to mappings."
        )

_SETTINGS_CLASSES: dict[str, type] = {
    "config": RunSettings,
    "flops_config": FlopsSettings,
}


def list_model_configs() -> list[str]:
    """Names accepted by ``--model``, e.g. ``timm/resnet50``."""
    model_dir = CONF_DIR / "model"
    # as_posix(): the name is a CLI identifier, so it must be "/"-separated on
    # every platform.  str(Path) yields "torchgeo\\scalemae_large_fmow" on
    # Windows, which no documented command or config would match.
    return sorted(
        p.relative_to(model_dir).as_posix().removesuffix(".yaml") for p in model_dir.rglob("*.yaml")
    )


def model_config_path(name: str) -> Path:
    """Return the YAML file backing ``--model <name>``."""
    if name not in list_model_configs():
        raise ValueError(f"Unknown model config '{name}'. {_closest_models(name)}")
    return CONF_DIR / "model" / f"{name}.yaml"


def _closest_models(name: str, n: int = 5) -> str:
    """Suggestion fragment naming the closest config names, or '' if none are close."""
    import difflib

    candidates = list_model_configs()
    matches = difflib.get_close_matches(name, candidates, n=n, cutoff=0.5)
    if not matches:
        # Fall back to substring hits — "resnet50" should still find its variants.
        matches = [c for c in candidates if name.lower() in c.lower()][:n]
    return f"Did you mean: {', '.join(matches)}? " if matches else ""


def compose_config(
    overrides: dict | None = None,
    *,
    config_name: str = "config",
    default_model: str | None = "rcf",
    model: str | None = None,
    config_path: str | Path | None = None,
) -> RunSettings | FlopsSettings:
    """Build the run/flops settings: defaults -> ``--config`` YAML -> model YAML -> CLI overrides.

    Args:
        overrides: Nested mapping of explicit CLI-flag overrides (e.g.
            ``{"dataset": {"batch_size": 8}}``), applied last so flags win.
        config_name: Which settings dataclass to start from (``"config"`` for
            :class:`~torchgeo_bench.settings.RunSettings`, ``"flops_config"``
            for ``FlopsSettings``).
        default_model: Model selected when no ``model`` name is given;
            ``None`` makes the model selection mandatory.
        model: Model name selected explicitly (e.g. from ``--model``).
        config_path: Optional YAML file of uncommon settings, merged over the
            defaults before the model YAML and CLI overrides are applied.

    Returns:
        The composed settings instance.
    """
    settings_cls = _SETTINGS_CLASSES.get(config_name)
    if settings_cls is None:
        raise ValueError(
            f"Unknown config_name {config_name!r}; expected one of {tuple(_SETTINGS_CLASSES)}"
        )
    cfg = settings_cls()

    if config_path is not None:
        user_settings = load_yaml(config_path)
        if not isinstance(user_settings, dict):
            raise ValueError(f"--config {config_path} must contain a YAML mapping at the top level")
        cfg = merge(cfg, user_settings)

    model_name = model or default_model
    if model_name is None:
        raise ValueError("No model selected; pass --model/-m (see `run --list-models`).")
    model_path = CONF_DIR / "model" / f"{model_name}.yaml"
    if not model_path.is_file():
        raise ValueError(
            f"Unknown model config {model_name!r}. "
            f"{_closest_models(model_name)}Run `torchgeo-bench run --list-models` for all "
            f"{len(list_model_configs())} configs."
        )
    cfg.model = load_yaml(model_path)
    if not isinstance(cfg.model, dict):
        raise ModelConfigError(f"{model_path}: model config must contain a YAML mapping.")
    validate_model_config(cfg.model, source=model_path)

    if overrides:
        cfg = merge(cfg, overrides)

    _resolve_seed_interpolation(cfg)
    return cfg


def _resolve_seed_interpolation(cfg: RunSettings | FlopsSettings) -> None:
    """Resolve ``model.seed: ${seed}`` (rcf.yaml) against the top-level ``seed``.

    This is the only interpolation used anywhere in the shipped configs, so
    it is resolved explicitly rather than adding general ``${...}`` support.
    Runs after CLI overrides so a ``--seed`` flag is reflected here too.
    """
    if cfg.model.get("seed") == "${seed}":
        cfg.model["seed"] = cfg.seed


def instantiate(config: dict, **kwargs: Any) -> Any:
    """Instantiate the class named by ``config["_target_"]`` with the remaining keys.

    Extra ``kwargs`` override same-named config keys. ``config`` is always a
    plain dict (a model config, or a nested ``_target_`` block like
    ``eval.segmentation.criterion``), so no conversion is needed.
    """
    conf = dict(config)
    target = conf.pop("_target_")
    module_name, _, attr = target.rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    conf.update(kwargs)
    return cls(**conf)
