"""Per-model results storage.

Each run writes to ``results/models/<model name>.csv`` instead of one shared
CSV, so adding or re-running a model touches only that model's file.  Use
:func:`load_results` to read the whole benchmark back as a single DataFrame.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = "results/models"

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str) -> str:
    """Return ``name`` reduced to characters that are safe in a filename."""
    cleaned = _UNSAFE.sub("_", str(name)).strip("._")
    if not cleaned:
        raise ValueError(f"model name {name!r} has no filename-safe characters")
    return cleaned


def model_results_path(results_dir: str | Path, model_name: str) -> Path:
    """Return the CSV path holding ``model_name``'s results."""
    return Path(results_dir) / f"{sanitize_name(model_name)}.csv"


def load_results(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    *,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every per-model CSV under ``results_dir``.

    Args:
        results_dir: Directory holding ``<model name>.csv`` files.
        names: Restrict to these model names; ``None`` loads all.

    Returns:
        One DataFrame of all rows, or an empty one if nothing is stored yet.
    """
    directory = Path(results_dir)
    if names is not None:
        paths = [model_results_path(directory, n) for n in names]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(directory.glob("*.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        logger.warning("No results found under %s", directory)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
