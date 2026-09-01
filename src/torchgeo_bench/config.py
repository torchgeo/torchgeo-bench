"""Config composition and ``_target_`` instantiation (replaces Hydra).

The benchmark's config system is one base YAML (``conf/config.yaml`` or
``conf/flops_config.yaml``), one model YAML selected from ``conf/model/``,
and ``key=value`` dotlist overrides.  Hydra provided that plus a lot of
machinery this project never used (multirun, output dirs, plugins, its own
override grammar); OmegaConf alone covers the actual needs and keeps CLI
startup fast.

Override semantics kept from the Hydra days so existing scripts still work:

* ``model=timm/resnet50`` selects ``conf/model/timm/resnet50.yaml``.
* ``dataset.names=[m-eurosat,m-so2sat]`` — values parse as YAML.
* Unknown keys are rejected (struct mode); prefix with ``+`` to force-add
  a new key (``+model.gsd=10``).
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
    # as_posix(): the name is a CLI identifier, so it must be "/"-separated on
    # every platform.  str(Path) yields "torchgeo\\scalemae_large_fmow" on
    # Windows, which no documented command or config would match.
    return sorted(
        p.relative_to(model_dir).as_posix().removesuffix(".yaml") for p in model_dir.rglob("*.yaml")
    )


def model_config_path(name: str) -> Path:
    """Return the YAML file backing ``model=<name>`` / ``--model <name>``."""
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
    overrides: Sequence[str] = (),
    *,
    config_name: str = "config",
    default_model: str | None = "rcf",
) -> DictConfig:
    """Build the run config: base YAML + model YAML + dotlist overrides.

    Args:
        overrides: ``key=value`` strings.  ``model=<name>`` selects the model
            YAML; ``+key=value`` adds a key not present in the base config.
        config_name: Base YAML under ``conf/`` (``config`` or ``flops_config``).
        default_model: Model selected when no ``model=`` override is given;
            ``None`` makes the override mandatory.

    Returns:
        The merged, struct-mode config.
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
            # Hydra spelled "add" as `+key` and "add-or-override" as `++key`.
            # Stripping only one `+` turned `++model.pool=cls` into a literal
            # `+model` node instead of an override — a silent no-op on the real
            # key, so strip the whole prefix.
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

    Extra ``kwargs`` override same-named config keys.  Nested values are
    passed as plain Python containers.  (Covers what this repo used from
    ``hydra.utils.instantiate``: flat configs, no nested targets.)
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)  # type: ignore[assignment]
    conf = dict(config)
    target = conf.pop("_target_")
    module_name, _, attr = target.rpartition(".")
    cls = getattr(importlib.import_module(module_name), attr)
    conf.update(kwargs)
    return cls(**conf)
