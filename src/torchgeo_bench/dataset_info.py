"""Dataset metadata definitions and config loading for torchgeo-bench.

Each dataset is described by a YAML file in ``conf/dataset/`` containing
task type, band information, sensor, and normalization statistics.
"""

from __future__ import annotations

import importlib.resources
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


@dataclass
class BandInfo:
    """Metadata for a single spectral band.

    Attributes:
        name: Full band name as it appears in the data files.
        short: Short human-friendly identifier (e.g. ``"red"``, ``"nir"``).
        wavelength_um: Approximate centre wavelength in micrometres (null for
            non-optical bands like SAR or DEM).
        mean: Dataset-level mean pixel value.
        std: Dataset-level standard deviation.
    """

    name: str
    short: str
    wavelength_um: float | None = None
    mean: float | None = None
    std: float | None = None


@dataclass
class DatasetInfo:
    """Structured metadata for a benchmark dataset.

    Loaded from a YAML config file in ``conf/dataset/``.

    Attributes:
        name: Dataset identifier used on the command line (e.g. ``"m-eurosat"``).
        task: ``"classification"`` or ``"segmentation"``.
        num_classes: Number of output classes.
        sensor: Sensor family (e.g. ``"sentinel-2"``, ``"aerial"``, ``"mixed"``).
        version: Dataset version tag — ``"v1"`` (GeoBench V1) or ``"v2"``
            (GeoBench V2).
        bands: Ordered list of all bands in the dataset.
        rgb_bands: Short names of the bands to use for RGB-only mode.
        multilabel: Whether labels are multi-hot (e.g. BigEarthNet).
        v2_class: For V2 datasets, the class name in ``geobench_v2.datasets``
            (e.g. ``"GeoBenchBENV2"``).
    """

    name: str
    task: str
    num_classes: int
    sensor: str = "unknown"
    version: str = "v1"
    bands: list[BandInfo] = field(default_factory=list)
    rgb_bands: list[str] = field(default_factory=lambda: ["red", "green", "blue"])
    multilabel: bool = False
    v2_class: str | None = None

    # ----- convenience helpers -----

    @property
    def num_channels(self) -> int:
        """Total number of bands."""
        return len(self.bands)

    @property
    def rgb_indices(self) -> list[int]:
        """Indices into ``bands`` for the RGB subset."""
        short_names = [b.short for b in self.bands]
        return [short_names.index(s) for s in self.rgb_bands if s in short_names]

    @property
    def is_v2(self) -> bool:
        """Whether this is a GeoBench V2 dataset."""
        return self.version == "v2"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file into a plain dict via OmegaConf."""
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _dict_to_dataset_info(d: dict) -> DatasetInfo:
    """Convert a raw YAML dict into a ``DatasetInfo``."""
    bands = [BandInfo(**b) for b in d.get("bands", [])]
    return DatasetInfo(
        name=d["name"],
        task=d["task"],
        num_classes=d["num_classes"],
        sensor=d.get("sensor", "unknown"),
        version=d.get("version", "v1"),
        bands=bands,
        rgb_bands=d.get("rgb_bands", ["red", "green", "blue"]),
        multilabel=d.get("multilabel", False),
        v2_class=d.get("v2_class"),
    )


def _conf_dataset_dir() -> Path:
    """Return the path to the packaged ``conf/dataset/`` directory."""
    pkg = importlib.resources.files("torchgeo_bench") / "conf" / "dataset"
    # importlib.resources may return a Traversable; cast to Path for os calls
    return Path(str(pkg))


@lru_cache(maxsize=64)
def load_dataset_info(name: str) -> DatasetInfo:
    """Load dataset metadata from its YAML config.

    Looks for ``conf/dataset/{safe_name}.yaml`` where *safe_name* is the
    dataset name with hyphens replaced by underscores.

    Args:
        name: Dataset identifier (e.g. ``"m-eurosat"``).

    Returns:
        A ``DatasetInfo`` instance.

    Raises:
        FileNotFoundError: If no config file exists for the dataset.
    """
    safe = name.replace("-", "_")
    yaml_name = f"{safe}.yaml"

    # Check CWD conf/dataset/ first
    cwd_path = Path("conf") / "dataset" / yaml_name
    if cwd_path.is_file():
        return _dict_to_dataset_info(_load_yaml(cwd_path))

    # Packaged config
    pkg_path = _conf_dataset_dir() / yaml_name
    if pkg_path.is_file():
        return _dict_to_dataset_info(_load_yaml(pkg_path))

    raise FileNotFoundError(
        f"No dataset config found for '{name}'. Looked in {cwd_path} and {pkg_path}."
    )


def list_available_datasets() -> list[str]:
    """Return names of all datasets that have config files."""
    names: set[str] = set()

    # Packaged configs
    pkg_dir = _conf_dataset_dir()
    if pkg_dir.is_dir():
        for p in pkg_dir.glob("*.yaml"):
            info = _dict_to_dataset_info(_load_yaml(p))
            names.add(info.name)

    # CWD overrides
    cwd_dir = Path("conf") / "dataset"
    if cwd_dir.is_dir():
        for p in cwd_dir.glob("*.yaml"):
            info = _dict_to_dataset_info(_load_yaml(p))
            names.add(info.name)

    return sorted(names)
