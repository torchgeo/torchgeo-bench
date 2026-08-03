"""Tiered leakage-safe fold assignment for the label-quality pipeline.

Each train sample is placed into one of ``k`` out-of-fold groups so that a
member never predicts on a sample it was trained on, *and* so that leaking
"twins" (same scene, near-duplicate tiles) cannot straddle the train/holdout
boundary. The tier used is recorded per dataset via a cascade:

1. ``native_id`` — a stable scene/AOI/tile ID recovered from the sample dict.
2. ``latlon_block`` — coarse lat/lon grid cells as indivisible groups.
3. ``phash`` — perceptual-hash near-duplicate groups (only when duplicates
   are actually found).
4. ``leakage_uncontrolled`` — plain random :class:`~sklearn.model_selection.KFold`
   last resort.

The grouped tiers wrap :class:`~sklearn.model_selection.GroupKFold`; the random
tier wraps :class:`~sklearn.model_selection.KFold`.
"""

import logging

import imagehash
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupKFold, KFold

logger = logging.getLogger(__name__)

# Candidate keys under which an upstream sample dict may expose a stable ID.
_NATIVE_ID_KEYS: tuple[str, ...] = ("native_id", "scene_id", "aoi_id", "tile_id")
# Large prime for a collision-resistant (lat_cell, lon_cell) -> scalar hash.
_CELL_HASH_PRIME = 100003


def assign_folds(
    dataset,
    split: str,
    k: int,
    *,
    cell_deg: float = 1.0,
    seed: int = 0,
    hamming_threshold: int = 4,
) -> tuple[np.ndarray, str]:
    """Assign each train sample a leakage-safe fold and report the tier used.

    Args:
        dataset: A :class:`~torchgeo_bench.datasets.base.BenchDataset` (anything
            exposing ``get_dataset(split, metadata=...)``).
        split: Split to fold (typically ``"train"``).
        k: Number of folds.
        cell_deg: lat/lon grid-cell size in degrees for the spatial-block tier.
        seed: RNG seed for the random fallback tier.
        hamming_threshold: Max perceptual-hash Hamming distance for two tiles to
            count as near-duplicates.

    Returns:
        ``(fold_ids, tier)`` where ``fold_ids`` has shape ``(N,)`` with values in
        ``range(k)`` and ``tier`` is one of ``"native_id"``, ``"latlon_block"``,
        ``"phash"``, ``"leakage_uncontrolled"``.
    """
    ds = dataset.get_dataset(split, metadata=["lat", "lon"])

    # Fast path: the native_id and latlon_block tiers only need per-sample
    # metadata, which V2 datasets already carry as columns on the upstream
    # ``data_df``. Reading it there avoids decoding every image off disk (a full
    # serial pass over the split — minutes for a few-thousand-sample dataset).
    # These yield byte-identical fold ids to the sample-materializing path.
    cheap = _cheap_fold_keys(ds)
    if cheap is not None:
        native, lat, lon = cheap
        if native is not None:
            return _group_kfold_ids(native, k), "native_id"
        if lat is not None and lon is not None:
            return _group_kfold_ids(_grid_cells(lat, lon, cell_deg), k), "latlon_block"

    # Slow path: no cheap metadata (or only phash discriminates). The phash tier
    # genuinely needs pixels, so here we do materialize every sample.
    samples = [ds[i] for i in range(len(ds))]
    n = len(samples)

    native = _native_ids(samples)
    if native is not None:
        return _group_kfold_ids(native, k), "native_id"

    latlon = _lat_lon(samples)
    if latlon is not None:
        lat, lon = latlon
        return _group_kfold_ids(_grid_cells(lat, lon, cell_deg), k), "latlon_block"

    groups = _phash_groups(samples, hamming_threshold)
    if groups is not None and _has_duplicate_group(groups):
        return _group_kfold_ids(groups, k), "phash"

    return _random_kfold_ids(n, k, seed), "leakage_uncontrolled"


def _cheap_fold_keys(
    ds,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
    """Per-sample fold keys read from the upstream ``data_df`` without image I/O.

    Unwraps the framework's dataset adapter down to the loader carrying the
    tortilla ``data_df`` and returns ``(native_ids, lat, lon)`` — any of which
    may be ``None`` if that column is absent. Returns ``None`` when there is no
    ``data_df`` at all (e.g. the synthetic test datasets), so the caller falls
    back to materializing samples. ``native_id`` takes precedence over lat/lon,
    matching the tier cascade.
    """
    inner = ds
    for _ in range(3):
        if hasattr(inner, "data_df"):
            break
        if hasattr(inner, "_inner"):
            inner = inner._inner
        else:
            break
    df = getattr(inner, "data_df", None)
    if df is None:
        return None

    cols = getattr(df, "columns", [])
    native = None
    for key in ("native_id", "scene_id", "aoi_id", "tile_id"):
        if key in cols:
            native = np.array([str(x) for x in df[key].tolist()])
            break

    lat = lon = None
    if "lat" in cols and "lon" in cols:
        lat_arr = np.asarray(df["lat"].tolist(), dtype=float)
        lon_arr = np.asarray(df["lon"].tolist(), dtype=float)
        if np.isfinite(lat_arr).all() and np.isfinite(lon_arr).all():
            lat, lon = lat_arr, lon_arr

    if native is None and lat is None:
        return None
    return native, lat, lon


def _native_ids(samples: list[dict]) -> np.ndarray | None:
    """Return per-sample native IDs if a single ID key is present on all samples."""
    for key in _NATIVE_ID_KEYS:
        if all(key in s and s[key] is not None for s in samples):
            return np.array([str(s[key]) for s in samples])
    return None


def _lat_lon(samples: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(lat, lon)`` arrays if every sample carries finite coordinates."""
    if not all("lat" in s and "lon" in s for s in samples):
        return None
    lat = np.array([float(s["lat"]) for s in samples])
    lon = np.array([float(s["lon"]) for s in samples])
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        return None
    return lat, lon


def _grid_cells(lat: np.ndarray, lon: np.ndarray, cell_deg: float) -> np.ndarray:
    """Map each coordinate to an integer grid-cell id (blockCV-style)."""
    return np.floor(lat / cell_deg).astype(np.int64) * _CELL_HASH_PRIME + np.floor(
        lon / cell_deg
    ).astype(np.int64)


def _phash_groups(samples: list[dict], hamming_threshold: int) -> np.ndarray | None:
    """Group samples by perceptual-hash near-duplicate via single-link union-find."""
    if not all("image" in s for s in samples):
        return None
    hashes = [_perceptual_hash(s["image"]) for s in samples]
    n = len(hashes)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if (hashes[i] - hashes[j]) <= hamming_threshold:
                parent[find(i)] = find(j)

    return np.array([find(i) for i in range(n)])


def _perceptual_hash(image) -> imagehash.ImageHash:
    """Perceptual hash of a ``(C, H, W)`` tensor after min-max scaling to 8-bit RGB."""
    arr = image[:3] if image.shape[0] >= 3 else image[:1].repeat(3, 1, 1)
    arr = arr.float()
    lo, hi = arr.min(), arr.max()
    arr = (arr - lo) / (hi - lo + 1e-8) * 255.0
    rgb = arr.byte().permute(1, 2, 0).cpu().numpy()
    return imagehash.phash(Image.fromarray(rgb))


def _has_duplicate_group(groups: np.ndarray) -> bool:
    """True if any group holds more than one sample."""
    _, counts = np.unique(groups, return_counts=True)
    return bool(counts.max() > 1)


def _group_kfold_ids(groups: np.ndarray, k: int) -> np.ndarray:
    """Per-sample fold ids from :class:`GroupKFold` keeping each group intact."""
    fold_ids = np.empty(len(groups), dtype=np.int64)
    splitter = GroupKFold(n_splits=k)
    for fold_idx, (_, test_idx) in enumerate(
        splitter.split(np.zeros((len(groups), 1)), groups=groups)
    ):
        fold_ids[test_idx] = fold_idx
    return fold_ids


def _random_kfold_ids(n: int, k: int, seed: int) -> np.ndarray:
    """Per-sample fold ids from a shuffled random :class:`KFold`."""
    fold_ids = np.empty(n, dtype=np.int64)
    splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_idx, (_, test_idx) in enumerate(splitter.split(np.arange(n))):
        fold_ids[test_idx] = fold_idx
    return fold_ids
