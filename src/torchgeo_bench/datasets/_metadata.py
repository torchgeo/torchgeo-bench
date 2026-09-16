"""Data-only GeoBench V1 metadata decoding."""

import json
import math


def decode_metadata(value: object) -> dict:
    """Decode JSON metadata containing band names and numeric classification labels."""
    if not isinstance(value, (str, bytes)):
        raise TypeError(f"Expected JSON metadata text or bytes, got {type(value).__name__}.")
    metadata = json.loads(value)
    if not isinstance(metadata, dict):
        raise TypeError("GeoBench metadata must be a JSON object.")

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
