"""Exercise real V1 JSON/NPZ shards, acquisition order, and visible corrupt-data failures."""

import io
import json
import pickle
import tarfile
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.support.data import write_v1_sample
from torchgeo_bench.datasets import get_dataset_spec, load_split
from torchgeo_bench.datasets._metadata import decode_metadata
from torchgeo_bench.datasets._v1_webdataset import GeoBenchv1Sharded
from torchgeo_bench.geography import _v1_shard_origins, extract_geography

RED = "04 - Red_2020-01-01"
GREEN = "03 - Green_2020-01-01"
SID = "47.5_-121.25_sample"
EXECUTED: list[bool] = []


def _metadata(label: int | list[int] = 1) -> dict:
    return {
        "label": label,
        "bands_order": [RED, GREEN],
        RED: {"transform": [10, 0, 500000, 0, -10, 5200000], "crs": "EPSG:32631"},
    }


def _record_execution() -> dict:
    EXECUTED.append(True)
    return _metadata()


class _ObjectMetadata:
    def __reduce__(self) -> tuple:
        return _record_execution, ()


@pytest.fixture(autouse=True)
def _no_object_execution() -> Iterator[None]:
    EXECUTED.clear()
    yield
    assert not EXECUTED


def _payload(encoding: str) -> bytes | str | np.void:
    raw = pickle.dumps(_ObjectMetadata(), protocol=4)
    if encoding == "bytes":
        return raw
    if encoding == "repr":
        return repr(raw)
    if encoding == "void":
        return np.void(raw)
    return f"__import__({__name__!r}, fromlist=['_record_execution'])._record_execution()"


def _write_partition(directory: Path, sample_ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "default_partition.json").write_text(json.dumps({"train": sample_ids}))


def _bands_npz(*, objects: bool = False) -> bytes:
    buffer = io.BytesIO()
    red = (
        np.array([_ObjectMetadata()], dtype=object)
        if objects
        else np.full((2, 3), 10, dtype=np.uint16)
    )
    np.savez(buffer, **{RED: red, GREEN: np.full((2, 3), 20, dtype=np.uint16)})
    return buffer.getvalue()


def _write_members(directory: Path, members: dict[str, bytes]) -> Path:
    _write_partition(directory, [SID])
    path = directory / "shard_00000.tar"
    with tarfile.open(path, "w") as shard:
        for suffix, payload in members.items():
            member = tarfile.TarInfo(f"{SID}.{suffix}")
            member.size = len(payload)
            shard.addfile(member, io.BytesIO(payload))
    return path


def _write_shard(directory: Path, metadata: bytes, suffix: str = "meta.json") -> Path:
    return _write_members(directory, {"bands.npz": _bands_npz(), suffix: metadata})


@pytest.mark.parametrize("as_bytes", [False, True])
def test_metadata_accepts_json_text_and_bytes(*, as_bytes: bool) -> None:
    metadata = _metadata()
    payload = json.dumps(metadata)
    assert decode_metadata(payload.encode() if as_bytes else payload) == metadata


@pytest.mark.parametrize(
    ("selection", "error", "message"),
    [
        ({"dataset_name": "missing", "split": "train"}, FileNotFoundError, "dir"),
        (
            {"dataset_name": "m-eurosat", "partition": "missing", "split": "train"},
            FileNotFoundError,
            r"missing_partition\.json",
        ),
        (
            {"dataset_name": "m-eurosat", "split": "invalid"},
            ValueError,
            "Split 'invalid' not found",
        ),
    ],
)
def test_reader_rejects_invalid_dataset_partition_and_split(
    tmp_path: Path, selection: dict[str, str], error: type[Exception], message: str
) -> None:
    _write_shard(tmp_path / "m-eurosat", json.dumps(_metadata()).encode())
    with pytest.raises(error, match=message):
        GeoBenchv1Sharded(root=tmp_path, **selection)


def test_v1_rejects_unknown_band_without_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="unknown band"):
        load_split("m-eurosat", "train", bands=("nonexistent_band",))


@pytest.mark.parametrize("label", [2, [1, 0, 1]])
@pytest.mark.parametrize("bands", [None, ("03 - Green", "04 - Red")])
def test_shards_preserve_raw_values_labels_and_dotted_sample_ids(
    tmp_path: Path, label: int | list[int], bands: tuple[str, ...] | None
) -> None:
    directory = tmp_path / "m-eurosat"
    ids = [SID + ".second", SID]
    _write_partition(directory, ids)
    arrays = {
        RED: np.full((2, 3), 10, np.uint16),
        GREEN: np.full((2, 3), 20, np.uint16),
    }
    # Reverse archive order: the partition, not tar traversal, owns sample order.
    for index, sid in enumerate(reversed(ids)):
        with tarfile.open(directory / f"shard_{index:05d}.tar", "w") as archive:
            write_v1_sample(archive, sid, arrays, _metadata(label))
    dataset = GeoBenchv1Sharded(directory.parent, directory.name, "train", bands=bands)
    expected_image = torch.tensor([10, 20], dtype=torch.float32).view(2, 1, 1).expand(2, 2, 3)
    if bands is not None:
        expected_image = expected_image.flip(0)
    assert dataset.sample_ids == ids
    for index, sid in enumerate(ids):
        sample = dataset[index]
        assert sample["sample_id"] == sid
        torch.testing.assert_close(sample["image"], expected_image)
        expected_label = torch.tensor(
            label, dtype=torch.float32 if isinstance(label, list) else torch.long
        )
        torch.testing.assert_close(sample["label"], expected_label)


@pytest.mark.parametrize("exact_match", [False, True])
def test_date_suffixed_bands_keep_first_npz_acquisition_and_custom_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, exact_match: bool
) -> None:
    spec = get_dataset_spec("m-forestnet")
    directory = tmp_path / spec.source.root / spec.storage_name
    ids = ["later-in-partition", SID]
    _write_partition(directory, ids)
    # Metadata order and chronological order must not replace NPZ's first prefix match.
    arrays = {
        "04 - Red_2022-01-01": np.full((2, 3), 22, np.uint16),
        "03 - Green_2021-01-01": np.full((2, 3), 31, np.uint16),
        RED: np.full((2, 3), 20, np.uint16),
        GREEN: np.full((2, 3), 30, np.uint16),
    }
    if exact_match:
        arrays["04 - Red"] = np.full((2, 3), 99, np.uint16)
    with tarfile.open(directory / "shard_00000.tar", "w") as archive:
        for sid in reversed(ids):
            write_v1_sample(
                archive, sid, arrays, {"label": 4, "bands_order": list(reversed(arrays))}
            )
    (directory / "small_partition.json").write_text(json.dumps({"train": [SID]}))
    monkeypatch.chdir(tmp_path)
    for partition, expected_ids in (("default", ids), ("small", [SID])):
        loaded = load_split("m-forestnet", "train", bands=("green", "red"), partition=partition)
        assert loaded.dataset.sample_ids == expected_ids
        assert loaded.bands == (spec.bands[1], spec.bands[2])
        assert all(
            a is b for a, b in zip(loaded.bands, (spec.bands[1], spec.bands[2]), strict=True)
        )
        expected = torch.tensor([31, 99 if exact_match else 22], dtype=torch.float32)
        for sample in loaded.dataset:
            torch.testing.assert_close(sample["image"], expected[:, None, None].expand(2, 2, 3))
            assert sample["label"].dtype == torch.long
            assert sample["label"].item() == 4


@pytest.mark.parametrize("label", [0.5, 1.0000000001])
def test_v1_scalar_labels_are_not_silently_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: float
) -> None:
    directory = tmp_path / "data/classification_v1.0_wds/m-eurosat"
    _write_shard(directory, json.dumps({**_metadata(), "label": label}).encode())
    monkeypatch.chdir(tmp_path)
    loaded = load_split("m-eurosat", "train", bands=("red", "green"), image_size=1)
    with pytest.raises(ValueError, match="integer values"):
        loaded.dataset[0]


@pytest.mark.parametrize("encoding", ["bytes", "repr", "void", "expression"])
def test_metadata_decoder_rejects_executable_representations(encoding: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        decode_metadata(_payload(encoding))


@pytest.mark.parametrize(
    "metadata",
    [
        {"label": 1},
        {"label": 1, "bands_order": "red"},
        {"label": 1, "bands_order": []},
        {"label": "class", "bands_order": ["red"]},
        {"label": [], "bands_order": ["red"]},
        {"label": [float("nan")], "bands_order": ["red"]},
    ],
)
def test_metadata_decoder_rejects_invalid_fields(metadata: dict) -> None:
    with pytest.raises(ValueError, match="GeoBench metadata"):
        decode_metadata(json.dumps(metadata))


@pytest.mark.parametrize("metadata", [[], None, 1, "text"])
def test_metadata_decoder_requires_json_object(metadata: object) -> None:
    with pytest.raises(TypeError, match="must be a JSON object"):
        decode_metadata(json.dumps(metadata))


@pytest.mark.parametrize("suffix", ["meta.pkl", "meta.json"])
@pytest.mark.parametrize("encoding", ["bytes", "repr", "expression"])
@pytest.mark.parametrize("bands", [None, ("04 - Red",)])
def test_sharded_reader_rejects_executable_metadata(
    tmp_path: Path, suffix: str, encoding: str, bands: tuple[str, ...] | None
) -> None:
    payload = _payload(encoding)
    assert isinstance(payload, bytes | str)
    source = tmp_path / "source"
    _write_shard(source, payload.encode() if isinstance(payload, str) else payload, suffix)
    with pytest.raises(ValueError, match=r"meta\.json|codec|Expecting value"):
        GeoBenchv1Sharded(source.parent, source.name, "train", bands=bands)[0]


@pytest.mark.parametrize("valid_json", [False, True])
def test_sharded_reader_never_falls_back_to_pickle(tmp_path: Path, *, valid_json: bool) -> None:
    source = tmp_path / "source"
    _write_members(
        source,
        {
            "bands.npz": _bands_npz(),
            "meta.json": json.dumps(_metadata()).encode() if valid_json else b"{",
            "meta.pkl": pickle.dumps(_ObjectMetadata()),
        },
    )
    if valid_json:
        assert GeoBenchv1Sharded(source.parent, source.name, "train")[0]["label"].item() == 1
    else:
        with pytest.raises(json.JSONDecodeError):
            GeoBenchv1Sharded(source.parent, source.name, "train")[0]


@pytest.mark.parametrize("cache", ["absent", "empty-shards", "old-hdf5"])
def test_only_shards_are_runtime_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cache: str
) -> None:
    if cache == "old-hdf5":
        directory = tmp_path / "data/classification_v1.0/m-eurosat"
        _write_partition(directory, [SID])
        (directory / f"{SID}.hdf5").write_bytes(b"unsupported old sample")
    elif cache == "empty-shards":
        _write_partition(tmp_path / "data/classification_v1.0_wds/m-eurosat", [SID])
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        FileNotFoundError, match="torchgeo-bench download geobench_v1 --datasets m-eurosat"
    ):
        load_split("m-eurosat", "train")


def test_public_loading_rejects_pickle_cache_with_replacement_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "data/classification_v1.0_wds/m-eurosat"
    _write_shard(directory, pickle.dumps(_ObjectMetadata()), "meta.pkl")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        ValueError, match="torchgeo-bench download geobench_v1 --datasets m-eurosat"
    ):
        load_split("m-eurosat", "train")


@pytest.mark.parametrize("missing", ["meta.json", "bands.npz", "sample"])
def test_selected_samples_require_both_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    members = {"bands.npz": _bands_npz(), "meta.json": json.dumps(_metadata()).encode()}
    if missing == "sample":
        members.clear()
    else:
        del members[missing]
    directory = tmp_path / "data/classification_v1.0_wds/m-eurosat"
    _write_members(directory, members)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=rf"{SID!r} requires"):
        load_split("m-eurosat", "train")


@pytest.mark.parametrize(
    "payload", [b"", b"not an npz archive", _bands_npz()[:-50], _bands_npz(objects=True)]
)
def test_sharded_reader_rejects_corrupt_or_object_arrays(tmp_path: Path, payload: bytes) -> None:
    directory = tmp_path / "source"
    _write_members(directory, {"bands.npz": payload, "meta.json": json.dumps(_metadata()).encode()})
    dataset = GeoBenchv1Sharded(directory.parent, directory.name, "train")
    with pytest.raises((EOFError, ValueError, zipfile.BadZipFile)):
        dataset[0]


@pytest.mark.parametrize("suffix", ["bands.npz", "meta.json"])
def test_sharded_reader_requires_regular_sample_members(tmp_path: Path, suffix: str) -> None:
    directory = tmp_path / "source"
    path = _write_shard(directory, json.dumps(_metadata()).encode())
    member = tarfile.TarInfo(f"{SID}.{suffix}")
    member.type = tarfile.DIRTYPE
    with tarfile.open(path, "a") as archive:
        archive.addfile(member)
    with pytest.raises(ValueError, match="Not a sample file"):
        GeoBenchv1Sharded(directory.parent, directory.name, "train")


def test_missing_band_is_not_substituted(tmp_path: Path) -> None:
    directory = tmp_path / "source"
    _write_shard(directory, json.dumps(_metadata()).encode())
    dataset = GeoBenchv1Sharded(directory.parent, directory.name, "train", bands=("05 - NIR",))
    with pytest.raises(KeyError, match="05 - NIR"):
        dataset[0]


@pytest.mark.parametrize("file", ["partition", "tar"])
def test_corrupt_local_files_fail_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str
) -> None:
    directory = tmp_path / "data/classification_v1.0_wds/m-eurosat"
    path = _write_shard(directory, json.dumps(_metadata()).encode())
    if file == "partition":
        path = directory / "default_partition.json"
    path.write_bytes(b"corrupt")
    monkeypatch.chdir(tmp_path)
    with pytest.raises((json.JSONDecodeError, tarfile.ReadError)):
        load_split("m-eurosat", "train")


@pytest.mark.parametrize("nine_coefficients", [False, True])
def test_geography_reads_json_affine_and_crs(tmp_path: Path, *, nine_coefficients: bool) -> None:
    metadata = _metadata()
    if nine_coefficients:
        metadata[RED]["transform"].extend([0, 0, 1])
    path = _write_shard(tmp_path, json.dumps(metadata).encode())
    assert _v1_shard_origins(str(path)) == [(500000.0, 5200000.0, "EPSG:32631")]


def test_geography_reports_missing_coordinates(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, json.dumps({"label": 1, "bands_order": [RED, GREEN]}).encode())
    assert _v1_shard_origins(str(path)) == [(None, None, "NOGEO")]


def test_geography_reports_invalid_crs_type(tmp_path: Path) -> None:
    metadata = _metadata()
    metadata[RED]["crs"] = 32631
    path = _write_shard(tmp_path, json.dumps(metadata).encode())
    assert _v1_shard_origins(str(path)) == [
        (None, None, "ERR TypeError: GeoBench JSON metadata CRS must be a string.")
    ]


def test_geography_reports_non_file_metadata_members(tmp_path: Path) -> None:
    path = tmp_path / "shard_00000.tar"
    member = tarfile.TarInfo("sample.meta.json")
    member.type = tarfile.DIRTYPE
    with tarfile.open(path, "w") as archive:
        archive.addfile(member)
    assert _v1_shard_origins(str(path)) == [
        (None, None, "ERR ValueError: Not a metadata file: sample.meta.json")
    ]


def test_geography_extracts_only_the_canonical_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "data/classification_v1.0_wds/m-eurosat"
    _write_shard(directory, json.dumps(_metadata()).encode())
    old_directory = tmp_path / "data/classification_v1.0/m-eurosat"
    old_directory.mkdir(parents=True)
    (old_directory / "unused.hdf5").write_bytes(b"unsupported")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("torchgeo_bench.geography._attribute_continents", lambda *_args: Counter())
    record = extract_geography("m-eurosat", workers=1)
    assert record.status == "extracted"
    assert record.n == 1
    assert record.points[0][0] == pytest.approx(3)
    assert 46 < record.points[0][1] < 47


def test_geography_reports_pickle_shards_as_errors(tmp_path: Path) -> None:
    path = _write_shard(tmp_path, pickle.dumps(_ObjectMetadata()), "meta.pkl")
    results = _v1_shard_origins(str(path))
    assert len(results) == 1
    assert results[0][:2] == (None, None)
    assert "ERR ValueError: missing .meta.json" in results[0][2]


@pytest.mark.parametrize("metadata", [None, b"{"])
def test_geography_extraction_reports_absent_or_corrupt_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: bytes | None
) -> None:
    directory = tmp_path / "data/classification_v1.0_wds/m-eurosat"
    payload = json.dumps({"label": 1, "bands_order": [RED]}).encode()
    _write_shard(directory, metadata if metadata is not None else payload)
    monkeypatch.chdir(tmp_path)
    if metadata is not None:
        with pytest.raises(RuntimeError, match=r"100\.0% of samples failed to read"):
            extract_geography("m-eurosat", workers=1)
    else:
        record = extract_geography("m-eurosat", workers=1)
        assert record.status == "no_geo"
        assert record.reason == "no sample carries a usable transform/crs"
