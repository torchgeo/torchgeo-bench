"""JSON metadata and checksum-approved GeoBench V1 pickle compatibility."""

import ast
import gzip
import hashlib
import io
import json
import math
import pickle
from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from typing import Literal, Self

import numpy as np

# Approved ForestNet records include polygon WKT and reach 169,322 bytes.
MAX_PICKLE_BYTES = 256 * 1024
_CHECKSUMS = files("torchgeo_bench.datasets").joinpath("_v1_checksums")


@cache
def _sources() -> dict:
    with _CHECKSUMS.joinpath("sources.json").open("r", encoding="utf-8") as stream:
        return json.load(stream)


def v1_source(layout: Literal["hdf5", "sharded"]) -> tuple[str, str]:
    """Return the repository and pinned revision covered by the bundled checksums."""
    source = _sources()[layout]
    return source["repo_id"], source["revision"]


@cache
def _checksums(dataset_name: str) -> dict[str, str]:
    if dataset_name not in _sources()["datasets"]:
        raise ValueError(f"No approved GeoBench V1 pickle metadata for dataset {dataset_name!r}.")
    with (
        _CHECKSUMS.joinpath(f"{dataset_name}.json.gz").open("rb") as stream,
        gzip.GzipFile(fileobj=stream) as uncompressed,
    ):
        return json.load(uncompressed)


class _BandInfo:
    """Inert stand-in for GeoBench's serialized band descriptions."""


class _LegacyAffine(tuple[float, ...]):
    """Read tuple-based Affine pickles independently of affine's current class layout."""

    def __new__(cls, a: float, b: float, c: float, d: float, e: float, f: float) -> Self:
        return super().__new__(cls, (a, b, c, d, e, f, 0.0, 0.0, 1.0))


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        if module == "geobench.dataset":
            return _BandInfo
        if (module, name) == ("affine", "Affine"):
            return _LegacyAffine
        return super().find_class(module, name)


def decode_metadata(value: object) -> dict:
    """Decode JSON metadata containing band names and numeric classification labels."""
    if not isinstance(value, (str, bytes)):
        raise TypeError(f"Expected JSON metadata text or bytes, got {type(value).__name__}.")
    return _validate_metadata(json.loads(value))


def _validate_metadata(metadata: object) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("GeoBench metadata must be a JSON object.")

    bands = metadata.get("bands_order")
    if (
        not isinstance(bands, list)
        or not bands
        or any(not isinstance(name, str) or not name for name in bands)
    ):
        raise ValueError("GeoBench metadata bands_order must be a non-empty list of band names.")

    label = metadata.get("label")
    labels = label if isinstance(label, list) else [label]
    if not labels or any(
        not isinstance(value, (int, float))
        or (isinstance(value, float) and not math.isfinite(value))
        for value in labels
    ):
        raise ValueError("GeoBench metadata label must be a number or a non-empty list of numbers.")
    return metadata


def decode_pickle_metadata(
    value: object,
    *,
    dataset_name: str,
    sample_id: str,
) -> dict:
    """Verify published metadata bytes before unpickling and return data-only fields.

    Trust comes only from the manifest shipped with this package, never from a
    checksum supplied alongside downloaded data. Unknown records fail closed.
    """
    expected = _checksums(dataset_name).get(sample_id)
    if expected is None:
        raise ValueError(
            f"No approved GeoBench V1 metadata for {dataset_name}/{sample_id}. "
            "Use the pinned release or supply data-only JSON metadata."
        )

    if isinstance(value, str):
        if len(value) > 4 * MAX_PICKLE_BYTES + 3:
            raise ValueError("GeoBench V1 pickle metadata exceeds the size limit.")
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("GeoBench V1 pickle metadata must be a bytes literal.") from exc
    elif isinstance(value, np.void):
        value = value.tobytes()
    if not isinstance(value, bytes):
        raise TypeError(f"Expected pickled metadata bytes, got {type(value).__name__}.")
    if len(value) > MAX_PICKLE_BYTES:
        raise ValueError("GeoBench V1 pickle metadata exceeds the size limit.")
    if hashlib.sha256(value).hexdigest() != expected:
        raise ValueError(
            f"GeoBench V1 metadata checksum mismatch for {dataset_name}/{sample_id}; "
            "refusing to unpickle. Use the pinned release or supply data-only JSON metadata."
        )

    # Unpickle these same immutable bytes, never reopen a path after verification.
    metadata = _MetadataUnpickler(io.BytesIO(value)).load()
    result = {
        "label": np.asarray(metadata["label"]).tolist(),
        "bands_order": list(metadata["bands_order"]),
    }
    for name, entry in metadata.items():
        if isinstance(entry, dict) and ("transform" in entry or "crs" in entry):
            transform, crs = entry.get("transform"), entry.get("crs")
            result[name] = {
                "transform": list(transform) if transform is not None else None,
                "crs": str(crs) if crs is not None else None,
            }
    return _validate_metadata(result)


def read_hdf5_metadata(attrs: Mapping[str, object], *, dataset_name: str, sample_id: str) -> dict:
    """Read JSON or checksum-approved pickle metadata from one HDF5 sample."""
    if "metadata_json" in attrs:
        return decode_metadata(attrs["metadata_json"])
    if "pickle" in attrs:
        return decode_pickle_metadata(
            attrs["pickle"], dataset_name=dataset_name, sample_id=sample_id
        )
    raise ValueError("GeoBench V1 sample has neither 'metadata_json' nor 'pickle' metadata.")
