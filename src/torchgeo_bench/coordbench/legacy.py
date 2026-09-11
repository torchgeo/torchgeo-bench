"""Temporary boundary for the unmigrated ``run mode=coord`` caller."""

from collections.abc import Callable
from functools import wraps
from typing import Any

from torchgeo_bench.coordbench.config import CoordConfig


def legacy_coord_config(config: Any) -> CoordConfig:
    """Translate only the legacy coordinate fields to the typed runtime schema."""
    from omegaconf import DictConfig, OmegaConf

    if not isinstance(config, DictConfig):
        raise TypeError("Coordinate evaluation requires a CoordConfig")
    model = OmegaConf.to_container(config.model, resolve=True)
    coord = OmegaConf.to_container(config.coord, resolve=True)
    if not isinstance(model, dict) or not isinstance(coord, dict):
        raise TypeError("Legacy model and coord sections must be mappings")
    target = model.pop("_target_")
    name = model.pop("name", str(target).rsplit(".", 1)[-1])
    names = coord.pop("names", "all")
    if isinstance(names, str):
        names = [name.strip() for name in names.split(",") if name.strip()]
    output = coord.pop("output", "results/coordbench_results.csv")
    if coord.get("knn_device") is None:
        coord["knn_device"] = "cpu"
    return CoordConfig.model_validate(
        {
            "model": {"name": name, "target": target, "kwargs": model},
            "datasets": names,
            "evaluation": coord,
            "runtime": {"seed": config.seed, "device": config.device},
            "output": {"file": output, "resume": config.resume},
        }
    )


def accepts_legacy_config(
    function: Callable[[CoordConfig], None],
) -> Callable[[CoordConfig], None]:
    """Keep the legacy caller outside the strictly typed coordinate runtime."""

    @wraps(function)
    def wrapped(config: CoordConfig) -> None:
        if not isinstance(config, CoordConfig):
            config = legacy_coord_config(config)
        function(config)

    return wrapped
