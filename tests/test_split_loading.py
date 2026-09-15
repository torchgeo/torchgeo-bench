"""Single-split ownership, validation, and caller-owned batching regressions."""

import io
import json
import tarfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from tests.support.data import write_classification_files
from tests.support.runner import _compose_cfg, _synthetic_splits
from torchgeo_bench.commands import _profile_runtime
from torchgeo_bench.config.profile import ProfileConfig
from torchgeo_bench.datasets import get_bench_dataset_class, load_split
from torchgeo_bench.datasets._v1_webdataset import GeoBenchv1Sharded
from torchgeo_bench.datasets.geobench_v1 import GeoBenchv1
from torchgeo_bench.main import run_dataset
from torchgeo_bench.resume import ResumeState


def _write_train_only(root: Path, dataset_name: str, storage: str) -> None:
    bench = get_bench_dataset_class(dataset_name)()
    family = "classification_v1.0_wds" if storage == "shards" else "classification_v1.0"
    directory = root / "data" / family / dataset_name
    directory.mkdir(parents=True)
    ids = [f"sample-{index}" for index in range(4)]
    (directory / "default_partition.json").write_text(json.dumps({"train": ids}))
    label = [int(index in (0, 5)) for index in range(bench.num_classes)] if bench.multilabel else 2
    metadata = json.dumps({"label": label, "bands_order": [b.source_name for b in bench.bands]})
    arrays = {
        band.source_name: np.full((4, 4), 1000 + index, dtype=np.float32)
        for index, band in enumerate(bench.bands)
    }
    if storage == "hdf5":
        for sample_id in ids:
            with h5py.File(directory / f"{sample_id}.hdf5", "w") as sample:
                for name, pixels in arrays.items():
                    sample[name] = pixels
                sample.attrs["metadata_json"] = metadata
    else:
        pixels = io.BytesIO()
        np.savez(pixels, **arrays)
        with tarfile.open(directory / "shard_00000.tar", "w") as archive:
            for sample_id in ids:
                for suffix, payload in (
                    ("meta.json", metadata.encode()),
                    ("bands.npz", pixels.getvalue()),
                ):
                    member = tarfile.TarInfo(f"{sample_id}.{suffix}")
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))


@pytest.fixture(params=["hdf5", "shards"])
def train_only(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> str:
    storage = request.param
    for name in ("m-eurosat", "m-bigearthnet"):
        _write_train_only(tmp_path, name, storage)
    monkeypatch.chdir(tmp_path)
    return storage


def test_train_only_constructs_no_unrelated_splits_loaders_or_sample_probes(
    train_only: str,
) -> None:
    bench_cls = get_bench_dataset_class("m-eurosat")
    backend = GeoBenchv1 if train_only == "hdf5" else GeoBenchv1Sharded
    bench = bench_cls()
    state = torch.get_rng_state()
    with (
        mock.patch.object(bench_cls, "_load_split", wraps=bench._load_split) as source,
        mock.patch.object(
            bench_cls, "resolve_band_specs", wraps=bench.resolve_band_specs
        ) as select,
        mock.patch.object(backend, "__getitem__", side_effect=AssertionError("sample probe")),
        mock.patch("torch.utils.data.DataLoader", side_effect=AssertionError("batching")),
    ):
        train = load_split("m-eurosat", "train", bands=("nir", "red"))
    source.assert_called_once()
    assert source.call_args.args == ("train",)
    assert source.call_args.kwargs["inputs"] is train.input
    select.assert_called_once_with(("nir", "red"))
    assert train.bands == (bench.bands[7], bench.bands[3])
    assert all(band is bench.bands[index] for band, index in zip(train.bands, (7, 3), strict=True))
    torch.testing.assert_close(torch.get_rng_state(), state)
    assert len(train.dataset) == 4
    assert train.dataset_name == "m-eurosat"
    assert train.split == "train"
    assert train.partition == "default"
    assert train.input.layout == "CHW"
    assert train.target_key == "label"
    assert train.num_classes == 10
    assert not train.multilabel
    for split in ("val", "test"):
        with pytest.raises(FileNotFoundError, match=f"split '{split}'"):
            load_split("m-eurosat", split)


@pytest.mark.parametrize("dataset_name", ["m-eurosat", "m-bigearthnet"])
@pytest.mark.parametrize("selection", ["rgb", "all", None, ("nir", "red"), ("red", "red")])
def test_ordered_metadata_and_raw_targets_are_preserved(
    train_only: str, dataset_name: str, selection: str | tuple[str, ...] | None
) -> None:
    loaded = load_split(dataset_name, "train", bands=selection, image_size=8)
    bench = get_bench_dataset_class(dataset_name)()
    sample = loaded.dataset[0]
    expected = torch.tensor([1000 + bench.bands.index(band) for band in loaded.bands]).float()
    torch.testing.assert_close(
        sample["image"], expected[:, None, None].expand(len(loaded.bands), 8, 8)
    )
    assert sample["image"].dtype == torch.float32
    assert sample["sample_id"] == "sample-0"
    if loaded.multilabel:
        assert sample["label"].dtype == torch.float32
        expected_label = torch.zeros(loaded.num_classes)
        expected_label[[0, 5]] = 1
        torch.testing.assert_close(sample["label"], expected_label)
    else:
        assert sample["label"].dtype == torch.long
        assert sample["label"].item() == 2
    if selection == "rgb":
        assert [band.name for band in loaded.bands] == bench.rgb_bands
    elif selection is None or selection == "all":
        assert loaded.bands == tuple(bench.bands)
    else:
        assert [band.name for band in loaded.bands] == list(selection)


def test_resolved_metadata_is_immutable(train_only: str) -> None:
    names = ["nir", "red"]
    loaded = load_split("m-eurosat", "train", bands=iter(names))
    names.reverse()
    assert loaded.input.selection == ("nir", "red")
    with pytest.raises(FrozenInstanceError):
        loaded.input.bands = ()
    with pytest.raises(FrozenInstanceError):
        loaded.dataset_name = "different"
    with pytest.raises(FrozenInstanceError):
        loaded.bands[0].mean = 0


@pytest.mark.parametrize("dataset_name", ["m-eurosat", "burn_scars", "eurosat", "resisc45"])
@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({"split": "validation"}, ValueError, "Unknown split"),
        ({"bands": "default"}, ValueError, "Unknown band selection"),
        ({"bands": "red"}, ValueError, "Unknown band selection"),
        ({"bands": ("not-a-band",)}, ValueError, "unknown band"),
        ({"bands": ()}, ValueError, "at least one channel"),
        ({"bands": 1}, TypeError, "ordered iterable"),
        ({"bands": {"red"}}, TypeError, "ordered iterable"),
        ({"bands": {"red": 1}}, TypeError, "ordered iterable"),
        ({"bands": ("red", 1)}, TypeError, "must be strings"),
        ({"partition": None}, TypeError, "partition"),
        ({"partition": ""}, ValueError, "partition"),
        ({"time_steps": 1}, ValueError, "not multi-temporal"),
        ({"time_steps": 0}, ValueError, "positive"),
        ({"time_steps": True}, TypeError, "integer"),
        ({"time_steps": 1.5}, TypeError, "integer"),
        ({"image_size": 0}, ValueError, "positive"),
        ({"image_size": True}, TypeError, "integer"),
        ({"interpolation": "wrong"}, ValueError, "interpolation"),
    ],
)
def test_invalid_options_fail_before_source_or_filesystem_access(
    dataset_name: str, options: dict, error: type[Exception], message: str
) -> None:
    bench_cls = get_bench_dataset_class(dataset_name)
    with (
        mock.patch.object(
            bench_cls, "_load_split", side_effect=AssertionError("constructed")
        ) as source,
        mock.patch.object(Path, "exists", side_effect=AssertionError("data access")),
        pytest.raises(error, match=message),
    ):
        load_split(dataset_name, **{"split": "train", **options})
    source.assert_not_called()


@pytest.mark.parametrize("dataset_name", ["burn_scars", "pastis", "eurosat", "resisc45"])
def test_unsupported_partition_is_never_ignored(dataset_name: str) -> None:
    bench_cls = get_bench_dataset_class(dataset_name)
    with (
        mock.patch.object(bench_cls, "_load_split") as source,
        pytest.raises(ValueError, match="does not support custom partitions"),
    ):
        load_split(dataset_name, "train", partition="small")
    source.assert_not_called()


def test_profile_reads_only_its_bounded_training_batch(
    train_only: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = GeoBenchv1 if train_only == "hdf5" else GeoBenchv1Sharded
    original = backend.__getitem__
    indices = []

    def read(self, index: int) -> dict:
        indices.append(index)
        return original(self, index)

    monkeypatch.setattr(backend, "__getitem__", read)
    config = ProfileConfig.model_validate(
        {
            "model": {"name": "rcf"},
            "dataset": "m-eurosat",
            "input": {"bands": ["nir", "red"], "image_size": 8},
            "runtime": {"batch_size": 2, "workers": 0},
        }
    )
    with mock.patch.object(_profile_runtime, "load_split", wraps=load_split) as load:
        train, batch = _profile_runtime._load_batch(config)
    assert load.call_count == 1
    assert load.call_args.args == ("m-eurosat", "train")
    assert len(indices) == 2
    assert batch.shape == (2, 2, 8, 8)
    assert [band.name for band in train.bands] == ["nir", "red"]


def test_image_runner_owns_batching_partition_policy_and_model_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = write_classification_files(tmp_path, "m-eurosat", (2, 7), all_bands=True)
    partition = json.loads((directory / "default_partition.json").read_text())
    (directory / "small_partition.json").write_text(json.dumps({"train": partition["train"][:4]}))
    monkeypatch.chdir(tmp_path)
    cfg = _compose_cfg(
        tmp_path / "unused.csv",
        {
            "model": {"name": "rcf", "kwargs": {"mode": "empirical"}},
            "input": {"partition": "small", "bands": ["nir", "red"]},
            "classification": {"methods": ["linear"]},
        },
    )
    bench_cls = get_bench_dataset_class("m-eurosat")
    captures = {}

    def build(preset, **kwargs):
        captures["model"] = kwargs
        return torch.nn.Identity()

    def evaluate(config, plan, model, loaders, metadata, **kwargs):
        captures["loaders"] = loaders
        return []

    with (
        mock.patch("torchgeo_bench.main.load_split", wraps=load_split) as load,
        mock.patch.object(
            bench_cls, "resolve_band_specs", wraps=bench_cls().resolve_band_specs
        ) as select,
        mock.patch("torchgeo_bench.main.build_model", side_effect=build),
        mock.patch("torchgeo_bench.main.run_classification", side_effect=evaluate),
        mock.patch.object(
            GeoBenchv1, "__getitem__", autospec=True, wraps=GeoBenchv1.__getitem__
        ) as probe,
    ):
        probe.side_effect = lambda self, index: {
            "image": torch.zeros(2, 8, 8),
            "label": torch.tensor(2),
        }
        assert list(run_dataset(cfg, "m-eurosat", ResumeState(set(), {}))) == []
    assert [call.args[1] for call in load.call_args_list] == ["train", "val", "test"]
    assert [call.kwargs["partition"] for call in load.call_args_list] == [
        "small",
        "default",
        "default",
    ]
    assert select.call_count == 3
    probe.assert_called_once()
    loaders = captures["loaders"]
    assert len(loaders.train.dataset) == 4
    assert len(loaders.val.dataset) == len(loaders.test.dataset) == 8
    assert isinstance(loaders.train.sampler, RandomSampler)
    assert isinstance(loaders.val.sampler, SequentialSampler)
    assert isinstance(loaders.test.sampler, SequentialSampler)
    for loader in (loaders.train, loaders.val, loaders.test):
        assert loader.batch_size == cfg.runtime.batch_size
        assert loader.num_workers == cfg.runtime.workers
        assert loader.pin_memory == torch.cuda.is_available()
        assert loader.generator is None
    assert captures["model"]["dataset"] is loaders.train.dataset
    expected = [bench_cls.bands[7], bench_cls.bands[3]]
    assert all(a is b for a, b in zip(captures["model"]["bands"], expected, strict=True))


def test_runner_rejects_split_input_disagreement_before_model_construction(tmp_path: Path) -> None:
    train, val, test = _synthetic_splits()
    val = replace(val, input=replace(val.input, bands=tuple(reversed(val.bands))))
    cfg = _compose_cfg(tmp_path / "unused.csv", {"classification": {"methods": ["linear"]}})
    with (
        mock.patch("torchgeo_bench.main.load_split", side_effect=[train, val, test]),
        mock.patch("torchgeo_bench.main.build_model") as build,
        pytest.raises(ValueError, match="metadata disagrees"),
    ):
        list(run_dataset(cfg, "m-eurosat", ResumeState(set(), {})))
    build.assert_not_called()


def test_gallery_loads_only_the_requested_split(train_only: str) -> None:
    from projects.cleanlab.render_flagged_gallery import _load_gallery_split

    loaded = _load_gallery_split("m-eurosat", "train")
    assert len(loaded.dataset) == 4
    assert loaded.dataset[0]["image"].shape == (len(loaded.bands), 4, 4)


def test_gallery_uses_resolved_channel_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from projects.cleanlab import render_flagged_gallery as gallery

    train = _synthetic_splits()[0]
    train = replace(train, input=replace(train.input, bands=tuple(reversed(train.bands))))
    monkeypatch.setattr(gallery, "_load_gallery_split", lambda *args: train)
    flagged = pd.DataFrame([{"index": 0, "given_label": 0, "guessed_label": 1, "issue_score": 0.9}])
    output = tmp_path / "gallery.png"
    with mock.patch.object(gallery, "_to_rgb", return_value=np.zeros((8, 8, 3))) as rgb:
        gallery.render_gallery("m-eurosat", "train", flagged, output, cols=1)
    assert rgb.call_args.args[1] == [2, 1, 0]
    assert output.is_file()


def test_explicit_partition_applies_to_requested_validation_split(train_only: str) -> None:
    family = "classification_v1.0_wds" if train_only == "shards" else "classification_v1.0"
    partition = Path("data") / family / "m-eurosat" / "validation-only_partition.json"
    partition.write_text(json.dumps({"valid": ["sample-3"]}))
    loaded = load_split("m-eurosat", "val", partition="validation-only")
    assert loaded.partition == "validation-only"
    assert loaded.dataset[0]["sample_id"] == "sample-3"
