"""Integration tests asserting train/val/test sample counts.

Expected sizes are hardcoded values derived from the *reference*
implementations:

- GeoBench V1: ``classification_v1.0/<dataset>/default_partition.json`` (the
  partition file is the source of truth that ``GeoBenchv1`` reads from).
- GeoBench V2: each ``geobench_v2.datasets.GeoBench<X>`` upstream class.
- EuroSAT (torchgeo template): ``torchgeo.datasets.EuroSAT``.

The test instantiates the registered :class:`BenchDataset` wrapper, builds a
``Dataset`` for each split via ``get_dataset(split)``, and checks ``len(ds)``
against the expected value.  Datasets whose data is not present on disk are
skipped (we catch ``FileNotFoundError`` and ``DatasetNotFoundError`` from the
upstream loader).

If a dataset's actual size on disk diverges from the hardcoded value, the
test will surface the mismatch — that may indicate either upstream changes
or a bug in our wrapper's ``get_dataset`` plumbing.
"""

import pytest
from torchgeo.datasets.errors import DatasetNotFoundError

from torchgeo_bench.datasets import (
    get_bench_dataset_class,
    list_datasets,
)

# Expected sizes per (dataset, split).  Sourced from:
#   * V1: ``data/classification_v1.0/<name>/default_partition.json``
#   * V2: instantiating ``geobench_v2.datasets.GeoBench<X>`` directly and
#     reading ``len(...)`` for each split
#   * eurosat: ``torchgeo.datasets.EuroSAT.split_filenames`` line counts
#     (16200 / 5400 / 5400, totalling the canonical 27000-image set)
EXPECTED_SIZES: dict[str, dict[str, int]] = {
    # GeoBench V1 (default partition)
    "m-eurosat": {"train": 2000, "val": 1000, "test": 1000},
    "m-forestnet": {"train": 6464, "val": 989, "test": 993},
    "m-so2sat": {"train": 19992, "val": 986, "test": 986},
    "m-pv4ger": {"train": 11814, "val": 999, "test": 999},
    "m-brick-kiln": {"train": 15063, "val": 999, "test": 999},
    "m-bigearthnet": {"train": 20000, "val": 1000, "test": 1000},
    # GeoBench V2 classification
    "benv2": {"train": 20000, "val": 4000, "test": 4000},
    "treesatai": {"train": 4000, "val": 1000, "test": 2000},
    "so2sat": {"train": 19992, "val": 986, "test": 986},
    "forestnet": {"train": 6464, "val": 989, "test": 993},
    "caffe": {"train": 4000, "val": 1000, "test": 2000},
    # GeoBench V2 segmentation
    "burn_scars": {"train": 524, "val": 160, "test": 120},
    "cloudsen12": {"train": 4000, "val": 535, "test": 975},
    "dynamic_earthnet": {"train": 700, "val": 100, "test": 200},
    "flair2": {"train": 4049, "val": 1022, "test": 3022},
    "fotw": {"train": 4000, "val": 1000, "test": 2000},
    "kuro_siwo": {"train": 4000, "val": 1000, "test": 2000},
    "pastis": {"train": 1455, "val": 482, "test": 496},
    "spacenet2": {"train": 5186, "val": 1461, "test": 2961},
    "spacenet7": {"train": 3500, "val": 652, "test": 1152},
    # torchgeo template
    "advance": {"train": 3045, "val": 1015, "test": 1015},
    "eurosat": {"train": 16200, "val": 5400, "test": 5400},
    "eurosat-spatial": {"train": 16200, "val": 5400, "test": 5400},
    "resisc45": {"train": 18900, "val": 6300, "test": 6300},
}


def test_expected_sizes_cover_registry():
    """Sanity check: every registered dataset has hardcoded expectations."""
    missing = sorted(set(list_datasets()) - set(EXPECTED_SIZES))
    assert not missing, (
        f"EXPECTED_SIZES is missing entries for {missing}. Add them after "
        "verifying against the upstream reference implementation."
    )


@pytest.mark.parametrize("dataset_name", sorted(EXPECTED_SIZES))
def test_split_sizes(dataset_name):
    """Each split's ``len(get_dataset(split))`` matches the reference value."""
    bench_cls = get_bench_dataset_class(dataset_name)
    bench = bench_cls()
    expected = EXPECTED_SIZES[dataset_name]

    actual: dict[str, int] = {}
    try:
        for split in ("train", "val", "test"):
            ds = bench.get_dataset(split, bands=tuple(bench.rgb_bands))
            actual[split] = len(ds)
    except (FileNotFoundError, DatasetNotFoundError) as exc:
        pytest.skip(f"{dataset_name}: data not found on disk ({exc})")

    assert actual == expected, (
        f"{dataset_name}: split sizes diverge from reference. expected={expected}, got={actual}"
    )


@pytest.mark.parametrize("dataset_name", sorted(EXPECTED_SIZES))
def test_declared_split_sizes_match_reference(dataset_name):
    """Each wrapper's ``BenchDataset.split_sizes`` matches the reference."""
    bench_cls = get_bench_dataset_class(dataset_name)
    declared = dict(bench_cls.split_sizes)
    expected = EXPECTED_SIZES[dataset_name]
    assert declared == expected, (
        f"{dataset_name}.split_sizes is out of sync with the reference. "
        f"declared={declared}, expected={expected}"
    )
