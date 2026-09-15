"""Check reference split sizes against available datasets; skip missing data."""

import json
from pathlib import Path

import pytest

from tests.support.data import require_dataset_data
from torchgeo_bench.datasets import get_dataset_spec, load_split

# V1 counts come from data/classification_v1.0/<name>/default_partition.json.
# V2 counts come from len(...) on each upstream geobench_v2.datasets.GeoBench<X> split.
# EuroSAT uses torchgeo split-file counts: 27000 images split 60/20/20.
# RESISC45 uses torchgeo split-file counts: 31500 images split 60/20/20.
EXPECTED_SIZES: dict[str, dict[str, int]] = {
    name: record["split_sizes"]
    for name, record in json.loads(
        (Path(__file__).parent / "fixtures/dataset_metadata.json").read_text()
    )["datasets"].items()
}


@pytest.mark.slow
@pytest.mark.parametrize("dataset_name", sorted(EXPECTED_SIZES))
def test_split_sizes(dataset_name: str) -> None:
    require_dataset_data(dataset_name)
    bench = get_dataset_spec(dataset_name)
    expected = EXPECTED_SIZES[dataset_name]

    actual = {
        split: len(load_split(dataset_name, split).dataset) for split in ("train", "val", "test")
    }

    assert dict(bench.split_sizes) == expected
    assert actual == expected, (
        f"{dataset_name}: split sizes diverge from reference. expected={expected}, got={actual}"
    )
