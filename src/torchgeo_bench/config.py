"""YAML configuration and ``_target_`` model construction.

Runs combine a base YAML (``conf/config.yaml`` or ``conf/flops_config.yaml``), a model YAML from ``conf/model/``, and ``key=value`` overrides.

Supported overrides:

* ``model=timm/resnet50`` selects ``conf/model/timm/resnet50.yaml``.
* ``dataset.names=[m-eurosat,m-so2sat]`` — values parse as YAML.
* Unknown keys are rejected; prefix with ``+`` to add a new key (``+model.gsd=10``).
"""

import importlib
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

CONF_DIR = Path(str(files("torchgeo_bench") / "conf"))


def list_model_configs() -> list[str]:
    """Names accepted by ``model=`` / ``--model``, e.g. ``timm/resnet50``."""
    model_dir = CONF_DIR / "model"
    # CLI model names use "/" on every platform, including Windows.
    return sorted(
        p.relative_to(model_dir).as_posix().removesuffix(".yaml") for p in model_dir.rglob("*.yaml")
    )


def model_config_path(name: str) -> Path:
    """Return the YAML file backing ``model=<name>`` / ``--model <name>``."""
    if name not in list_model_configs():
        raise ValueError(f"Unknown model config '{name}'. {_closest_models(name)}")
    return CONF_DIR / "model" / f"{name}.yaml"


def _closest_models(name: str, n: int = 5) -> str:
    """Suggest similar model names, or return an empty string if none match."""
    import difflib

    candidates = list_model_configs()
    matches = difflib.get_close_matches(name, candidates, n=n, cutoff=0.5)
    if not matches:
        # Fall back to substring hits — "resnet50" should still find its variants.
        matches = [c for c in candidates if name.lower() in c.lower()][:n]
    return f"Did you mean: {', '.join(matches)}? " if matches else ""


def compose_config(
    overrides: Sequence[str] = (),
    *,
    config_name: str = "config",
    default_model: str | None = "rcf",
) -> DictConfig:
    """Build the run config from base YAML, model YAML, and ``key=value`` overrides.

    Args:
        overrides: ``key=value`` strings.  ``model=<name>`` selects the model
            YAML; ``+key=value`` adds a key not present in the base config.
        config_name: Base YAML under ``conf/`` (``config`` or ``flops_config``).
        default_model: Model selected when no ``model=`` override is given;
            ``None`` makes the override mandatory.

    Returns:
        The merged config, which rejects unknown keys.
    """
    cfg = OmegaConf.load(CONF_DIR / f"{config_name}.yaml")
    assert isinstance(cfg, DictConfig)

    model_name = default_model
    dotlist: list[str] = []
    additions: list[str] = []
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Malformed override {override!r}: expected key=value")
        if key == "model":
            model_name = value
        elif key.startswith("+"):
            # Existing Hydra scripts use both +key (add) and ++key (add or override).
            additions.append(f"{key.lstrip('+')}={value}")
        else:
            dotlist.append(override)

    if model_name is None:
        raise ValueError("No model selected; pass --model/-m (see `run --list-models`).")
    model_path = CONF_DIR / "model" / f"{model_name}.yaml"
    if not model_path.is_file():
        raise ValueError(
            f"Unknown model config {model_name!r}. "
            f"{_closest_models(model_name)}Run `torchgeo-bench run --list-models` for all "
            f"{len(list_model_configs())} configs."
        )
    cfg.model = OmegaConf.load(model_path)

    OmegaConf.set_struct(cfg, True)
    if additions:
        with open_dict(cfg):
            cfg.merge_with(OmegaConf.from_dotlist(additions))
    if dotlist:
        cfg.merge_with(OmegaConf.from_dotlist(dotlist))
    return cfg


def instantiate(config: DictConfig | dict, **kwargs: Any) -> Any:
    """Instantiate the class named by ``config._target_`` with the remaining keys.

    Extra ``kwargs`` override config keys. Nested config values become plain Python containers; nested ``_target_`` values are not instantiated.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
    conf = dict(config)
    target = conf.pop("_target_")
    module_name, _, attr = target.rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    conf.update(kwargs)
    return cls(**conf)
