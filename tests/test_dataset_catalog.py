"""Parity, immutability, discovery, and import-isolation tests for dataset specs."""

import json
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, asdict, fields, is_dataclass
from pathlib import Path

import pytest
import yaml

from torchgeo_bench.cli import main
from torchgeo_bench.datasets import (
    BandSpec,
    DatasetSpec,
    SplitSizes,
    TorchGeoSource,
    V2Source,
    catalog,
    get_dataset_spec,
    get_dataset_task,
    list_datasets,
)

BASELINE = json.loads((Path(__file__).parent / "fixtures/dataset_metadata.json").read_text())


def _json_record(spec: DatasetSpec) -> dict:
    return json.loads(json.dumps(asdict(spec)))


@pytest.fixture(params=["classification", "segmentation"])
def registered_dataset(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> DatasetSpec:
    spec = DatasetSpec(
        name="catalog-test",
        task=request.param,
        num_classes=2,
        bands=(BandSpec("aerial", "gray", "gray", 0, 1, 0, 1),),
        default_bands=("gray",),
        split_sizes=SplitSizes(1, 1, 1),
        source=V2Source("UnavailableReader"),
    )
    monkeypatch.setattr(
        catalog, "_CATALOG", catalog._make_catalog((*catalog._CATALOG.values(), spec))
    )
    return spec


def test_registered_dataset_appears_in_catalog(
    registered_dataset: DatasetSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["datasets"])
    assert capsys.readouterr().out.splitlines() == list_datasets()
    assert registered_dataset.name in list_datasets()
    main(["datasets", registered_dataset.name])
    assert yaml.safe_load(capsys.readouterr().out) == _json_record(registered_dataset)
    assert get_dataset_spec(registered_dataset.name) is registered_dataset


@pytest.mark.parametrize("source", ["flags", "yaml"])
def test_dry_run_accepts_registered_dataset(
    registered_dataset: DatasetSpec,
    source: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if source == "flags":
        arguments = ["--model", "rcf", "--dataset", registered_dataset.name, "--device", "cpu"]
    else:
        config = tmp_path / "run.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "model": {"name": "rcf"},
                    "datasets": [registered_dataset.name],
                    "runtime": {"device": "cpu"},
                }
            ),
            encoding="utf-8",
        )
        arguments = ["--config", str(config)]
    main(["run", *arguments, "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out)["datasets"] == [registered_dataset.name]


def test_catalog_matches_frozen_baseline_coverage() -> None:
    assert BASELINE["baseline_commit"] == "97501cfe64216d9a34849d57ff10f5c9cb04a2c8"
    assert list_datasets() == sorted(BASELINE["datasets"])
    assert len(list_datasets()) == 23


@pytest.mark.parametrize("name", sorted(BASELINE["datasets"]))
def test_all_metadata_matches_pre_migration_definition(name: str) -> None:
    spec = get_dataset_spec(name)
    source = spec.source
    actual = {
        "name": spec.name,
        "task": spec.task,
        "num_classes": spec.num_classes,
        "multilabel": spec.multilabel,
        "bands": [asdict(band) for band in spec.bands],
        "rgb_bands": list(spec.rgb_bands) if spec.rgb_bands is not None else None,
        "default_bands": list(spec.default_bands),
        "split_sizes": dict(spec.split_sizes),
        "supports_partitions": spec.capabilities.supports_partitions,
        "multi_temporal": spec.capabilities.multi_temporal,
        "target_key": spec.target_key,
        "data_root": source.root,
        "source_kind": source.kind,
        "storage_name": spec.storage_name,
        "geography_alias": spec.geography.alias_of,
        "geography_reason": spec.geography.reason,
        "validation_split": source.validation_split,
    }
    if isinstance(source, V2Source):
        upstream_kwargs = {}
        if source.return_stacked_image is not None:
            upstream_kwargs["return_stacked_image"] = source.return_stacked_image
        if source.time_step is not None:
            upstream_kwargs["time_step"] = list(source.time_step)
        old_adapter = {
            "identity": "_V2Dataset.canonicalize_sample",
            "later_acquisition": "FieldsOfTheWorld.canonicalize_sample",
            "post_sar_dem": "KuroSiwo.canonicalize_sample",
            "offset_mask": "_OffsetMaskV2Dataset.canonicalize_sample",
        }[source.sample_adapter]
        actual.update(
            upstream_class=source.upstream_class,
            band_order_strategy=source.band_order_strategy,
            upstream_kwargs=upstream_kwargs,
            canonical_sensor_order=(
                list(source.canonical_sensor_order) if source.canonical_sensor_order else None
            ),
            sample_adapter=old_adapter,
        )
    if isinstance(source, TorchGeoSource):
        actual["upstream_class"] = source.upstream_class
    assert actual == BASELINE["datasets"][name]


@pytest.mark.parametrize("name", list_datasets())
def test_catalog_detail_and_task_use_complete_spec(
    name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = get_dataset_spec(name)
    assert get_dataset_task(name) == spec.task
    main(["datasets", name])
    assert yaml.safe_load(capsys.readouterr().out) == _json_record(spec)


def _assert_deeply_immutable(value: object) -> None:
    if is_dataclass(value):
        assert value.__dataclass_params__.frozen
        for field in fields(value):
            _assert_deeply_immutable(getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            _assert_deeply_immutable(item)
    else:
        assert value is None or isinstance(value, str | int | float | bool)


def test_catalog_and_nested_definitions_are_immutable() -> None:
    for name in list_datasets():
        spec = get_dataset_spec(name)
        _assert_deeply_immutable(spec)
        hash(spec)
    spec = get_dataset_spec("kuro_siwo")
    with pytest.raises(FrozenInstanceError):
        spec.name = "other"
    with pytest.raises(FrozenInstanceError):
        spec.source.time_step = ("pre_1",)
    with pytest.raises(FrozenInstanceError):
        spec.split_sizes.train = 3
    with pytest.raises(TypeError):
        catalog._CATALOG[spec.name] = spec
    with pytest.raises(ValueError, match="Duplicate dataset"):
        catalog._make_catalog((spec, spec))


def test_eurosat_identities_share_only_the_intended_metadata_and_storage() -> None:
    standard = get_dataset_spec("eurosat")
    spatial = get_dataset_spec("eurosat-spatial")
    v1 = get_dataset_spec("m-eurosat")
    assert len({standard.name, spatial.name, v1.name}) == 3
    assert standard.storage_name == spatial.storage_name == "eurosat"
    assert standard.source.root == spatial.source.root != v1.source.root
    assert spatial.bands is standard.bands
    assert v1.bands != standard.bands


def test_unknown_dataset_task_raises() -> None:
    with pytest.raises(KeyError, match="unregistered-dataset"):
        get_dataset_task("unregistered-dataset")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--help"],
        ["run", "--help"],
        ["datasets"],
        ["datasets", "m-eurosat"],
        ["datasets", "burn_scars"],
        ["datasets", "kuro_siwo"],
        ["models"],
        ["models", "rcf"],
        ["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu", "--dry-run"],
    ],
)
def test_full_lookup_and_discovery_block_heavy_imports(arguments: list[str]) -> None:
    code = """
import importlib.abc
import sys
from pathlib import Path
from unittest.mock import patch

blocked = {'torch', 'torchgeo', 'timm', 'numpy', 'pandas', 'geobench_v2', 'h5py'}
readers = {'torchgeo_bench.datasets.' + name for name in
           ('geobench_v1', 'geobench_v2', '_v1_webdataset', 'torchgeo')}
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked and fullname not in readers, fullname
sys.meta_path.insert(0, BlockImports())
from torchgeo_bench.datasets import get_dataset_spec, list_datasets
with patch.object(Path, 'exists', side_effect=AssertionError('data access')):
    for name in list_datasets():
        spec = get_dataset_spec(name)
        spec.validate_source()
        assert spec.bands and spec.split_sizes
from torchgeo_bench.download import DEFAULT_V2_DATASETS, DOWNLOADABLE_DATASETS
assert set(DOWNLOADABLE_DATASETS) == set(list_datasets())
assert 'caffe' in DEFAULT_V2_DATASETS and 'benv2' in DEFAULT_V2_DATASETS
from torchgeo_bench.cli import main
try:
    main(sys.argv[1:])
except SystemExit as error:  # allow-except: argparse help exits successfully
    assert error.code == 0
assert not blocked & sys.modules.keys()
assert not readers & sys.modules.keys()
assert not any(name.startswith('torchgeo_bench.models') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, *arguments],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
