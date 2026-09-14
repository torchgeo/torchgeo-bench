"""Lightweight discovery of packaged Pydantic model presets."""

import difflib
from importlib.resources import files
from pathlib import Path

CONF_DIR = Path(str(files("torchgeo_bench") / "conf"))


def list_model_configs() -> list[str]:
    """Return portable names accepted by ``--model`` and YAML ``model.name``."""
    model_dir = CONF_DIR / "model"
    return sorted(
        path.relative_to(model_dir).as_posix().removesuffix(".yaml")
        for path in model_dir.rglob("*.yaml")
    )


def model_config_path(name: str) -> Path:
    """Return a catalog-validated preset path, rejecting path traversal."""
    names = list_model_configs()
    if name not in names:
        matches = difflib.get_close_matches(name, names, n=5, cutoff=0.5)
        if not matches:
            matches = [candidate for candidate in names if name.lower() in candidate.lower()][:5]
        suggestion = f" Did you mean: {', '.join(matches)}?" if matches else ""
        raise ValueError(f"Unknown model config {name!r}.{suggestion}")
    return CONF_DIR / "model" / f"{name}.yaml"
