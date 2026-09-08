"""Check committed geography records without loading raw imagery."""

import json

import pytest

from torchgeo_bench.datasets import list_datasets
from torchgeo_bench.geography import (
    GEO_ALIAS,
    INDEX_NAME,
    NO_GEO,
    STORE_DIR,
    GeoRecord,
    list_geography,
    missing_datasets,
)

VALID_STATUSES = {"extracted", "no_geo", "not_downloaded"}

pytestmark = pytest.mark.skipif(
    not STORE_DIR.is_dir(),
    reason=f"geography store not generated at {STORE_DIR}",
)


@pytest.fixture(scope="module")
def store() -> dict[str, GeoRecord]:
    return list_geography()


def test_all_registered_datasets_have_a_record(store: dict[str, GeoRecord]) -> None:
    """Keep new datasets on the map; generate missing records with ``python experiments/scripts/extract_dataset_geography.py --all``."""
    assert missing_datasets() == set(), (
        f"registered datasets with no geography record: {sorted(missing_datasets())}"
    )


def test_no_extra_records(store: dict[str, GeoRecord]) -> None:
    assert set(store) <= set(list_datasets())


def test_statuses_are_valid(store: dict[str, GeoRecord]) -> None:
    for name, record in store.items():
        assert record.status in VALID_STATUSES, f"{name}: bad status {record.status!r}"


def test_absent_coordinates_are_explained(store: dict[str, GeoRecord]) -> None:
    """A dataset without coordinates must say why, so the map can disclose it."""
    for name, record in store.items():
        if record.status in ("no_geo", "not_downloaded"):
            assert record.reason, f"{name}: status={record.status} but no reason given"


def test_known_no_geo_datasets_are_declared(store: dict[str, GeoRecord]) -> None:
    """Datasets with no source coordinates must retain that status."""
    for name in NO_GEO:
        if name in store:
            assert store[name].status == "no_geo", f"{name} unexpectedly has coordinates"


def test_aliases_point_at_their_source(store: dict[str, GeoRecord]) -> None:
    """Re-split datasets borrow geometry from the dataset holding it."""
    for name, target in GEO_ALIAS.items():
        if name in store and store[name].status == "extracted":
            assert store[name].alias_of == target


def test_extracted_records_are_wellformed(store: dict[str, GeoRecord]) -> None:
    extracted = [r for r in store.values() if r.status == "extracted"]
    if not extracted:
        pytest.skip("no dataset has extracted coordinates in this store")

    for record in extracted:
        assert record.n > 0, f"{record.name}: extracted but n=0"
        assert record.bbox is not None, f"{record.name}: extracted but no bbox"

        min_lon, min_lat, max_lon, max_lat = record.bbox
        assert -180 <= min_lon <= max_lon <= 180, f"{record.name}: bad lon bbox {record.bbox}"
        assert -90 <= min_lat <= max_lat <= 90, f"{record.name}: bad lat bbox {record.bbox}"

        assert record.bins, f"{record.name}: no density bins"
        assert record.points, f"{record.name}: no sampled points"
        assert sum(c for _, _, c in record.bins) == record.n, (
            f"{record.name}: bin counts do not sum to n"
        )

        if record.continents:
            total = sum(record.continents.values())
            assert 99.0 <= total <= 101.0, f"{record.name}: continents sum to {total}"


def test_sampled_points_lie_within_bbox(store: dict[str, GeoRecord]) -> None:
    for record in store.values():
        if record.status != "extracted" or record.bbox is None:
            continue
        min_lon, min_lat, max_lon, max_lat = record.bbox
        # Bounds are rounded to three decimal places, so allow 0.001 at each edge.
        for lon, lat, *_ in record.points:
            assert min_lon - 0.001 <= lon <= max_lon + 0.001, (
                f"{record.name}: lon {lon} outside bbox"
            )
            assert min_lat - 0.001 <= lat <= max_lat + 0.001, (
                f"{record.name}: lat {lat} outside bbox"
            )


def test_index_matches_the_records(store: dict[str, GeoRecord]) -> None:
    index_path = STORE_DIR / INDEX_NAME
    assert index_path.exists(), "index.json missing; run the extractor to rebuild it"

    index = json.loads(index_path.read_text())
    entries = {e["name"]: e for e in index["datasets"]}
    assert set(entries) == set(store), "index.json is stale relative to the record files"

    for name, entry in entries.items():
        assert entry["status"] == store[name].status
        assert entry["n"] == store[name].n

    totals = index["totals"]
    assert totals["datasets"] == len(store)
    assert totals["samples"] == sum(r.n for r in store.values())


def test_records_roundtrip_through_json(store: dict[str, GeoRecord]) -> None:
    for record in store.values():
        assert GeoRecord.from_json(record.to_json()) == record


def test_stored_files_are_canonical_json() -> None:
    """Canonical JSON keeps regenerated records byte-identical."""
    for path in sorted(STORE_DIR.glob("*.json")):
        raw = path.read_text()
        assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":")), (
            f"{path.name} is not canonical; regenerate it with the extractor"
        )
