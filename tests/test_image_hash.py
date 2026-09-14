"""Full payloads captured from d5168ad, before either image interface was migrated."""

import gzip
import json
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.datasets import get_dataset_task
from torchgeo_bench.image_hash import (
    _hash_payload,
    _historical_payload,
    _resume_config_hash,
    _resume_config_payload,
    compatible_hashes,
)
from torchgeo_bench.main import main

from .test_main_fast import _resume_row

with gzip.open(Path(__file__).parent / "data/image-hash-baseline.json.gz", "rt") as file:
    BASELINE = json.load(file)

with (Path(__file__).parent / "fixtures/image-cli-hash-cases.json").open() as file:
    IMAGE_CLI_CASES = json.load(file)


def _config(name: str = "rcf", **sections) -> RunConfig:
    return RunConfig.model_validate(
        {
            "model": {"name": name},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            **sections,
        }
    )


@pytest.mark.parametrize("name", BASELINE)
@pytest.mark.parametrize("style", ["legacy", "image"])
def test_all_128_historical_payloads_and_hashes_are_exact(name: str, style: str) -> None:
    payload = _historical_payload(_config(name), style)
    assert payload == BASELINE[name][style]["payload"]
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        BASELINE[name][style]["payload"], sort_keys=True
    )
    assert _hash_payload(payload) == BASELINE[name][style]["hash"]


@pytest.mark.parametrize("name", BASELINE)
def test_default_canonical_hash_retains_equivalent_public_cli_payload(name: str) -> None:
    model = BASELINE[name]["legacy"]["payload"]["model"]
    changed_probe_defaults = "c_range" in model.get("eval", {})
    coordinate_encoder = ".coordbench." in model["_target_"]
    style = "legacy" if changed_probe_defaults or coordinate_encoder else "image"
    assert _resume_config_payload(_config(name)) == BASELINE[name][style]["payload"]
    assert _resume_config_hash(_config(name)) == BASELINE[name][style]["hash"]


@pytest.mark.parametrize("case", IMAGE_CLI_CASES, ids=lambda case: case["id"])
def test_canonical_hash_matches_captured_public_cli_cases(case: dict) -> None:
    config = RunConfig.model_validate(case["input"])
    assert _resume_config_hash(config) == case["hash"]
    for dataset in config.datasets:
        hashes = compatible_hashes(
            config, dataset, segmentation=get_dataset_task(dataset) == "segmentation"
        )
        assert case["first_typed_hash"] in hashes


def test_canonical_format_selection_does_not_depend_on_dataset_selection() -> None:
    hashes = set()
    for datasets in (["m-eurosat"], ["caffe"], ["m-eurosat", "caffe"], ["all"]):
        config = _config(
            model={"name": "torchgeo/scalemae_large_fmow", "kwargs": {"res": 9.0}},
            datasets=datasets,
        )
        hashes.add(_resume_config_hash(config))
        assert _hash_payload(_historical_payload(config, "image")) not in compatible_hashes(
            config, "m-eurosat", segmentation=False
        )
    assert len(hashes) == 1


def test_real_legacy_segmentation_fingerprint_remains_resumable() -> None:
    config = _config(
        model={"name": "timm/resnet18", "kwargs": {"pretrained": False}},
        datasets=["caffe"],
        input={"image_size": 32},
        runtime={"device": "cpu", "batch_size": 2, "workers": 0},
        classification={"bootstrap_samples": 2},
        segmentation={"head": "linear", "epochs": 1, "batch_size": 2, "cache_dtype": "float32"},
    )
    assert "17895d85c4b14519" in compatible_hashes(config, "caffe", segmentation=True)


@pytest.mark.parametrize("style", ["legacy", "image"])
def test_resume_accepts_both_existing_formats_without_loading_data(
    tmp_path: Path, style: str
) -> None:
    out = tmp_path / "out.csv"
    config = _config(output={"file": str(out), "resume": True})
    rows = [
        _resume_row(config, method=method, metric_name="accuracy") for method in ("knn5", "linear")
    ]
    for row in rows:
        row["config_hash"] = BASELINE["rcf"][style]["hash"]
    pd.DataFrame(rows).to_csv(out, index=False)
    with mock.patch("torchgeo_bench.main.get_datasets", side_effect=AssertionError("no loading")):
        main(config)
    assert len(pd.read_csv(out)) == 2


def test_incompatible_old_adapter_defaults_are_not_reused() -> None:
    config = _config("torchgeo/croma_large")
    aliases = compatible_hashes(config, "m-eurosat", segmentation=False)
    assert BASELINE[config.model.name]["legacy"]["hash"] in aliases
    assert BASELINE[config.model.name]["image"]["hash"] not in aliases


def test_explicit_values_do_not_collide_with_old_preset_precedence() -> None:
    original = _config("torchgeo/resnet50_s2rgb_satlas_si")
    config = _config(original.model.name, segmentation={"layers": [], "learning_rate": 0.02})
    assert _resume_config_hash(config) != _resume_config_hash(original)
    assert BASELINE[config.model.name]["legacy"]["hash"] not in compatible_hashes(
        config, "caffe", segmentation=True
    )


def test_hash_preserves_dataset_overrides_and_explicit_constructor_precedence() -> None:
    config = _config("torchgeo/scalemae_large_fmow")
    config.model.kwargs["res"] = 2.0
    payload = _resume_config_payload(config)
    assert payload["model"]["dataset_overrides"]["m-eurosat"]["res"] == 2.0
    assert "names" not in payload["dataset"]
    assert "eval" in payload["model"]


def test_integer_float_endpoints_keep_explicit_compatibility_aliases() -> None:
    aliases = compatible_hashes(_config(), "m-eurosat", segmentation=False)
    payload = _historical_payload(_config(), "legacy")
    payload["eval"]["c_range"][0] = -6.0
    assert _hash_payload(payload) in aliases


def test_custom_constructor_mappings_are_not_interpreted_as_metadata() -> None:
    config = _config(
        model={
            "name": "custom",
            "target": "custom.Model",
            "kwargs": {"eval": {"segmentation": "ordinary value"}, "image_size": 16},
        },
        segmentation={"layers": []},
        input={"image_size": None},
    )
    original = config.model_dump()
    payload = _resume_config_payload(config)
    assert payload["model"]["kwargs"] == config.model.kwargs
    assert config.model_dump() == original
    config.model.kwargs["image_size"] = 32
    assert _hash_payload(payload) != _resume_config_hash(config)


@pytest.mark.parametrize("field", ["layers", "learning_rate", "head", "cache_features"])
def test_explicit_segmentation_defaults_are_hashed(field: str) -> None:
    value = {"layers": [], "learning_rate": 0.001, "head": "fpn", "cache_features": False}[field]
    config = _config("olmoearth_v1_2_base", segmentation={field: value})
    assert (
        _resume_config_payload(config)["eval"]["segmentation"][
            {"head": "head_type", "learning_rate": "lr"}.get(field, field)
        ]
        == value
    )
