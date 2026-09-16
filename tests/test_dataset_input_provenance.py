"""Metadata-only preflight, authoritative reuse, and scientific input fingerprints."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.support.data import write_caffe_files
from tests.support.runner import _compose_cfg
from torchgeo_bench.commands import _profile_runtime
from torchgeo_bench.config.presets import resolve_run_config
from torchgeo_bench.config.profile import ProfileConfig
from torchgeo_bench.datasets import (
    DatasetSpec,
    get_dataset_spec,
    list_datasets,
    load_split,
    resolve_input,
)
from torchgeo_bench.datasets.input import DATASET_INPUT_PROTOCOL_VERSION
from torchgeo_bench.main import run_dataset
from torchgeo_bench.resume import ResumeState, resume_config_hash


@pytest.mark.parametrize("name", list_datasets())
def test_serializable_description_is_independent_and_preserves_raw_metadata(name: str) -> None:
    spec = get_dataset_spec(name)
    inputs = resolve_input(spec, bands="default")
    description = inputs.description
    assert description["protocol_version"] == DATASET_INPUT_PROTOCOL_VERSION == 2
    assert description["dataset"] == name
    assert description["source"]["root"] == spec.source.root
    assert description["source"]["storage_name"] == spec.storage_name
    assert description["split_policy"]["val"] == spec.source.validation_split
    assert description["selection"] == "default"
    assert description["bands"] == [asdict(band) for band in inputs.bands]
    assert inputs.band_names == spec.default_bands
    assert len(inputs.fingerprint) == 64
    encoded = json.dumps(description, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert inputs.fingerprint == hashlib.sha256(encoded.encode()).hexdigest()
    description["bands"][0]["mean"] = -99999
    assert inputs.description["bands"][0]["mean"] == inputs.bands[0].mean
    if spec.rgb_bands is not None:
        rgb = resolve_input(spec)
        assert rgb.selection == "rgb"
        assert rgb.bands == inputs.bands
        assert all(a is b for a, b in zip(rgb.bands, inputs.bands, strict=True))
        assert rgb.fingerprint != inputs.fingerprint


@pytest.mark.parametrize("name", ["caffe", "kuro_siwo"])
@pytest.mark.parametrize("options", [{}, {"bands": "rgb"}])
def test_rgb_fails_before_source_import_and_io(name: str, options: dict) -> None:
    code = """
import importlib.abc
import json
import sys
from pathlib import Path
from unittest.mock import patch

blocked = {'torch', 'torchgeo', 'geobench_v2', 'numpy', 'pandas'}
readers = {'torchgeo_bench.datasets.' + name for name in
           ('geobench_v1', 'geobench_v2', '_v1_webdataset', 'torchgeo')}
class BlockImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split('.')[0] not in blocked and fullname not in readers, fullname
sys.meta_path.insert(0, BlockImports())
from torchgeo_bench.datasets import load_split, resolve_input
with patch.object(Path, 'exists', side_effect=AssertionError('data access')):
    for call in (lambda: resolve_input(sys.argv[1], **json.loads(sys.argv[2])),
                 lambda: load_split(sys.argv[1], 'train', **json.loads(sys.argv[2]))):
        try:
            call()
        except ValueError as error:
            assert 'no genuine RGB' in str(error)
            assert all(choice in str(error) for choice in ('default', 'all', 'explicit'))
        else:
            raise AssertionError('accepted non-RGB input')
    inputs = resolve_input(sys.argv[1], bands='default')
    assert inputs.description['selection'] == 'default'
    assert len(inputs.fingerprint) == 64
"""
    result = subprocess.run(
        [sys.executable, "-c", code, name, json.dumps(options)],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("dataset", "inputs", "message"),
    [
        ("caffe", {}, "no genuine RGB"),
        ("kuro_siwo", {"bands": "rgb"}, "no genuine RGB"),
        ("caffe", {"bands": "default", "partition": "small"}, "custom partitions"),
        ("caffe", {"bands": "default", "time_steps": 2}, "not multi-temporal"),
        ("m-eurosat", {"bands": ["not-a-band"]}, "unknown band"),
    ],
)
def test_runner_preflight_precedes_resume_skip_and_loading(
    tmp_path: Path, dataset: str, inputs: dict, message: str
) -> None:
    config = _compose_cfg(
        tmp_path / "unused.csv",
        {"datasets": [dataset], "input": inputs, "output": {"resume": True}},
    )
    with (
        patch("torchgeo_bench.main.plan_dataset_run", side_effect=AssertionError("resume")),
        patch("torchgeo_bench.main.load_split", side_effect=AssertionError("source")),
        pytest.raises(ValueError, match=message),
    ):
        list(run_dataset(config, dataset, ResumeState(set(), {})))
    profile = ProfileConfig.model_validate(
        {"model": {"name": "rcf"}, "dataset": dataset, "input": inputs}
    )
    with (
        patch.object(_profile_runtime, "load_split", side_effect=AssertionError("source")),
        pytest.raises(ValueError, match=message),
    ):
        _profile_runtime._load_batch(profile)


def test_loaded_input_is_the_preflight_object_without_reselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_caffe_files(tmp_path)
    monkeypatch.chdir(tmp_path)
    inputs = resolve_input("caffe", bands="default")
    with patch.object(DatasetSpec, "resolve_band_specs", side_effect=AssertionError("reselected")):
        loaded = load_split("caffe", "train", bands="default", inputs=inputs)
    assert loaded.input is inputs
    assert loaded.input.description == inputs.description
    assert loaded.input.fingerprint == inputs.fingerprint
    assert loaded.dataset[0]["image"].shape[0] == len(inputs.bands) == 1


def test_preflight_input_type_is_checked_before_loading() -> None:
    with (
        patch("torchgeo_bench.datasets.loading._load_source", side_effect=AssertionError("source")),
        pytest.raises(TypeError, match="inputs must be a ResolvedInput"),
    ):
        load_split("m-eurosat", "train", inputs=1)


@pytest.mark.parametrize(
    "options",
    [
        {"bands": "all"},
        {"partition": "small"},
        {"time_steps": 1},
    ],
)
def test_conflicting_preflight_options_do_not_load_a_source(options: dict) -> None:
    inputs = resolve_input("m-eurosat")
    with (
        patch("torchgeo_bench.datasets.loading._load_source", side_effect=AssertionError("source")),
        pytest.raises(ValueError, match=r"Preflight inputs|not multi-temporal"),
    ):
        load_split("m-eurosat", "train", inputs=inputs, **options)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sensor", "different-sensor"),
        ("name", "renamed-channel"),
        ("source_name", "different-source-name"),
        ("mean", 123.456),
        ("std", 234.567),
        ("min", -100.0),
        ("max", 50000.0),
        ("wavelength_um", 0.123),
    ],
)
def test_every_selected_band_field_invalidates_input_and_resume_hash(
    tmp_path: Path, field: str, value: object
) -> None:
    cfg, model = resolve_run_config(_compose_cfg(tmp_path / "unused.csv"), "m-eurosat")
    spec = get_dataset_spec("m-eurosat")
    inputs = resolve_input(spec, bands="all")
    changed = replace(spec, bands=(replace(spec.bands[0], **{field: value}), *spec.bands[1:]))
    other = resolve_input(changed, bands="all")
    assert other.fingerprint != inputs.fingerprint
    assert resume_config_hash(cfg, model, inputs) != resume_config_hash(cfg, model, other)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root", "data/other-source"),
        ("storage_name", "other-archive"),
        ("revision", "another-pinned-revision"),
        ("repository", "another/mirror"),
        ("adapter_version", 2),
    ],
)
def test_source_identity_changes_invalidate(field: str, value: object) -> None:
    spec = get_dataset_spec("m-eurosat")
    changed = replace(spec, source=replace(spec.source, **{field: value}))
    assert resolve_input(changed).fingerprint != resolve_input(spec).fingerprint


def test_dataset_split_band_order_and_requested_semantics_invalidate() -> None:
    spec = get_dataset_spec("m-eurosat")
    base = resolve_input(spec, bands="all").fingerprint
    assert resolve_input(replace(spec, name="another-id"), bands="all").fingerprint != base
    assert resolve_input(replace(spec, bands=spec.bands[::-1]), bands="all").fingerprint != base
    assert resolve_input(spec, bands="all", partition="0.10x_train").fingerprint != base
    v2 = get_dataset_spec("burn_scars")
    changed = replace(v2, source=replace(v2.source, validation_split="val"))
    assert resolve_input(v2).fingerprint != resolve_input(changed).fingerprint
    assert resolve_input("eurosat").fingerprint != resolve_input("eurosat-spatial").fingerprint
    reduced = resolve_input("kuro_siwo", bands="default")
    explicit = resolve_input("kuro_siwo", bands=("vv", "vh"))
    assert reduced.bands == explicit.bands
    assert reduced.fingerprint != explicit.fingerprint
    assert explicit.fingerprint != resolve_input("kuro_siwo", bands=("vh", "vv")).fingerprint


def test_temporal_and_acquisition_policies_invalidate() -> None:
    temporal = [resolve_input("pastis", time_steps=steps) for steps in (None, 1, 2, 3)]
    assert len({inputs.fingerprint for inputs in temporal}) == 4
    assert [inputs.layout for inputs in temporal] == ["CHW", "CHW", "TCHW", "TCHW"]
    inputs = resolve_input("kuro_siwo", bands="default")
    assert inputs.description["source"]["time_step"] == ("post",)
    changed = replace(
        inputs, spec=replace(inputs.spec, source=replace(inputs.spec.source, time_step=("pre_1",)))
    )
    assert changed.fingerprint != inputs.fingerprint
    later = resolve_input("fotw")
    assert later.description["source"]["sample_adapter"] == "later_acquisition"
    changed = replace(
        later,
        spec=replace(later.spec, source=replace(later.spec.source, sample_adapter="identity")),
    )
    assert changed.fingerprint != later.fingerprint


def test_nonsemantic_unselected_metadata_does_not_invalidate_rgb() -> None:
    spec = get_dataset_spec("m-eurosat")
    changed = replace(
        spec,
        bands=(replace(spec.bands[0], mean=-1), *spec.bands[1:]),
        split_sizes=replace(spec.split_sizes, train=999),
        geography=replace(spec.geography, reason="not an image input"),
    )
    assert resolve_input(spec).fingerprint == resolve_input(changed).fingerprint
