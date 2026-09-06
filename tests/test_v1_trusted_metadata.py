"""Exercise the real bundled trust manifest without approving synthetic pickles."""

import hashlib
import io
import json
import pickle
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from rasterio.crs import CRS

from experiments.scripts.repack_geobench_v1 import repack, validate
from torchgeo_bench.datasets import _metadata, get_bench_dataset_class, list_datasets
from torchgeo_bench.datasets._v1_webdataset import GeoBenchv1Sharded
from torchgeo_bench.datasets.geobench_v1 import GeoBenchv1, _V1Dataset
from torchgeo_bench.geography import _v1_origin

from .test_cli_program import classification_arguments, run_cli

FIXTURES = Path(__file__).parent / "fixtures" / "v1_metadata"
S2_BANDS = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]
EUROSAT_TRANSFORM = (
    9.986548378452909,
    0,
    676723.1812297984,
    0,
    -9.980495333332907,
    5830903.856663333,
)
REFERENCES = [
    (
        "m-bigearthnet",
        "id_100699",
        "m-bigearthnet-0.pkl",
        [int(index in (8, 12, 27)) for index in range(43)],
        S2_BANDS[:10] + S2_BANDS[11:],
        (20, 0, 699960, 0, -20, 5620800),
        "EPSG:32631",
    ),
    (
        "m-brick-kiln",
        "examples_0_103",
        "m-brick-kiln-0.pkl",
        0,
        S2_BANDS,
        (8.983152841191e-05, 0, 88.53346282639944, 0, 8.983152841196551e-05, 26.47981929304796),
        "EPSG:4326",
    ),
    ("m-eurosat", "id_21603", "m-eurosat-0.pkl", 0, S2_BANDS, EUROSAT_TRANSFORM, "EPSG:32631"),
    ("m-eurosat", "id_22198", "m-eurosat-1.pkl", 1, S2_BANDS, EUROSAT_TRANSFORM, "EPSG:32631"),
    (
        "m-forestnet",
        "-1.6655276613404206_128.1211044868538_2016_01_01",
        "m-forestnet-0.pkl",
        3,
        [
            f"{band}_2016-01-01"
            for band in (
                "04 - Red",
                "03 - Green",
                "02 - Blue",
                "05 - NIR",
                "06 - SWIR1",
                "07 - SWIR2",
            )
        ],
        (
            0.00013480239144588455,
            0,
            128.09872728987378,
            0,
            -0.00013566292664363958,
            -1.6430076129066375,
        ),
        "EPSG:4326",
    ),
    (
        "m-pv4ger",
        "5.89550015,51.01899493",
        "m-pv4ger-0.pkl",
        0,
        ["Red", "Green", "Blue"],
        (
            1.4250829027728783e-06,
            0,
            5.8952721367355565,
            0,
            -8.988840608514792e-07,
            51.01913875144793,
        ),
        "EPSG:4326",
    ),
    (
        "m-so2sat",
        "id_0003",
        "m-so2sat-0.pkl",
        1,
        [
            "01 - VH.Real",
            "02 - VH.Imaginary",
            "03 - VV.Real",
            "04 - VV.Imaginary",
            "05 - VH.LEE Filtered",
            "06 - VV.LEE Filtered",
            "07 - VH.LEE Filtered.Real",
            "08 - VV.LEE Filtered.Imaginary",
            *S2_BANDS[1:9],
            *S2_BANDS[11:],
        ],
        None,
        None,
    ),
]
RECORDS = [
    {
        "dataset_name": dataset,
        "sample_id": sample_id,
        "filename": filename,
        "layout": layout,
        "metadata": {
            "label": label,
            "bands_order": list(bands),
            **{
                band: {
                    "transform": [*transform, 0, 0, 1] if transform is not None else None,
                    "crs": crs,
                }
                for band in bands
            },
        },
    }
    for dataset, sample_id, filename, label, bands, transform, crs in REFERENCES
    for layout in ("hdf5", "sharded")
]


@pytest.fixture(params=RECORDS, ids=lambda r: f"{r['dataset_name']}-{r['layout']}-{r['sample_id']}")
def record(request: pytest.FixtureRequest) -> dict:
    return request.param


def _decode(payload: object, record: dict) -> dict:
    return _metadata.decode_pickle_metadata(
        payload,
        dataset_name=record["dataset_name"],
        sample_id=record["sample_id"],
    )


def _payload(record: dict) -> bytes:
    return (FIXTURES / record["filename"]).read_bytes()


def _assert_metadata(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    assert actual["label"] == expected["label"]
    assert actual["bands_order"] == expected["bands_order"]
    for name, entry in expected.items():
        if not isinstance(entry, dict):
            continue
        transform, crs = entry["transform"], entry["crs"]
        if transform is None:
            assert actual[name]["transform"] is None
        else:
            assert actual[name]["transform"][:6] == pytest.approx(transform[:6])
        if crs is None:
            assert actual[name]["crs"] is None
        else:
            assert CRS.from_user_input(actual[name]["crs"]) == CRS.from_user_input(crs)
    assert _metadata.decode_metadata(json.dumps(actual, allow_nan=False)) == actual


def _write_source(
    root: Path,
    record: dict,
    *,
    payload: bytes | None = None,
    json_metadata: str | None = None,
) -> Path:
    directory = root / record["dataset_name"]
    directory.mkdir(parents=True)
    sid = record["sample_id"]
    (directory / "default_partition.json").write_text(
        json.dumps(dict.fromkeys(("train", "valid", "test"), [sid]))
    )
    bands = {
        name: np.full((8, 8), index + 100, dtype=np.float32)
        for index, name in enumerate(record["metadata"]["bands_order"])
    }
    raw = _payload(record) if payload is None else payload
    if record["layout"] == "hdf5":
        with h5py.File(directory / f"{sid}.hdf5", "w") as sample:
            for name, image in bands.items():
                sample.create_dataset(name, data=image)
            sample.attrs["pickle"] = repr(raw)
            if json_metadata is not None:
                sample.attrs["metadata_json"] = json_metadata
    else:
        buffer = io.BytesIO()
        np.savez(buffer, **bands)
        parts = {"bands.npz": buffer.getvalue(), "meta.pkl": raw}
        if json_metadata is not None:
            parts["meta.json"] = json_metadata.encode()
        with tarfile.open(directory / "shard_00000.tar", "w") as archive:
            for suffix, value in parts.items():
                member = tarfile.TarInfo(f"{sid}.{suffix}")
                member.size = len(value)
                archive.addfile(member, io.BytesIO(value))
    return directory


def _dataset(directory: Path, record: dict) -> GeoBenchv1 | GeoBenchv1Sharded:
    cls = GeoBenchv1 if record["layout"] == "hdf5" else GeoBenchv1Sharded
    return cls(directory.parent, directory.name, "train")


class _ExecutableMetadata:
    def __reduce__(self) -> tuple:
        return eval, ("1 / 0",)


def _forbid_unpickling(*args, **kwargs) -> None:
    pytest.fail("Unapproved metadata reached the unpickler.")


def test_manifest_and_fixtures_cover_both_published_layouts() -> None:
    datasets = {
        name for name in list_datasets() if issubclass(get_bench_dataset_class(name), _V1Dataset)
    }
    assert set(_metadata._sources()["datasets"]) == datasets
    assert {(r["dataset_name"], r["layout"]) for r in RECORDS} == {
        (name, layout) for name in datasets for layout in ("hdf5", "sharded")
    }
    for name in datasets:
        checksums = _metadata._checksums(name)
        assert checksums
        assert _metadata._sources()["audit"]["datasets"][name][
            "identical_raw_bytes_by_sample"
        ] == len(checksums)
        for sample_id, value in checksums.items():
            assert sample_id
            assert len(value) == 64
            assert set(value) <= set("0123456789abcdef")


@pytest.mark.parametrize("layout", ["hdf5", "sharded"])
def test_approved_metadata_cannot_be_swapped_between_samples(layout: str, monkeypatch) -> None:
    first, second = [
        r for r in RECORDS if r["dataset_name"] == "m-eurosat" and r["layout"] == layout
    ][:2]
    assert first["sample_id"] != second["sample_id"]
    assert first["metadata"]["label"] != second["metadata"]["label"]
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _decode(_payload(first), second)


@pytest.mark.parametrize("encoding", ["bytes", "repr", "void"])
def test_approved_reference_metadata_decodes(record: dict, encoding: str) -> None:
    raw = _payload(record)
    value = {"bytes": raw, "repr": repr(raw), "void": np.void(raw)}[encoding]
    _assert_metadata(_decode(value, record), record["metadata"])


def test_readers_load_approved_metadata(tmp_path: Path, record: dict) -> None:
    directory = _write_source(tmp_path, record)
    sample = _dataset(directory, record)[0]
    assert sample["sample_id"] == record["sample_id"]
    assert sample["label"].tolist() == record["metadata"]["label"]
    assert sample["image"].shape == (len(record["metadata"]["bands_order"]), 8, 8)


@pytest.mark.parametrize("change", ["replace", "truncate", "append", "benign"])
def test_checksum_rejection_precedes_unpickling(record: dict, monkeypatch, change: str) -> None:
    raw = _payload(record)
    candidates = {
        "replace": pickle.dumps(_ExecutableMetadata()),
        "truncate": raw[:-1],
        "append": raw + pickle.dumps(_ExecutableMetadata()),
        "benign": pickle.dumps({"label": 1, "bands_order": ["red"]}),
    }
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _decode(candidates[change], record)


@pytest.mark.parametrize(
    ("field", "value"),
    [("dataset_name", "../m-eurosat"), ("sample_id", "unapproved-sample")],
)
def test_unknown_record_cannot_borrow_an_approved_checksum(
    record: dict, monkeypatch, field: str, value: str
) -> None:
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    with pytest.raises(ValueError, match="No approved"):
        _decode(_payload(record), {**record, field: value})


@pytest.mark.parametrize("encoding", ["bytes", "repr", "void", "expression"])
def test_executable_representations_are_rejected_for_known_samples(
    record: dict, monkeypatch, encoding: str
) -> None:
    raw = pickle.dumps(_ExecutableMetadata())
    value = {
        "bytes": raw,
        "repr": repr(raw),
        "void": np.void(raw),
        "expression": "__import__('builtins').eval('1 / 0')",
    }[encoding]
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    with pytest.raises(ValueError):
        _decode(value, record)


@pytest.mark.parametrize("encoding", ["bytes", "repr", "void"])
def test_oversized_metadata_is_rejected(record: dict, monkeypatch, encoding: str) -> None:
    raw = bytes(_metadata.MAX_PICKLE_BYTES + 1)
    value = {"bytes": raw, "repr": repr(raw), "void": np.void(raw)}[encoding]
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    with pytest.raises(ValueError, match="size limit"):
        _decode(value, record)


def test_unpickler_receives_the_verified_copy(record: dict, monkeypatch) -> None:
    memory = bytearray(_payload(record))
    value = np.frombuffer(memory, dtype=f"V{len(memory)}")[0]
    original = _metadata._MetadataUnpickler

    def mutate_input(stream: io.BytesIO) -> pickle.Unpickler:
        memory[:] = bytes(len(memory))
        return original(stream)

    monkeypatch.setattr(_metadata, "_MetadataUnpickler", mutate_input)
    _assert_metadata(_decode(value, record), record["metadata"])
    assert memory == bytes(len(memory))


def test_dataset_supplied_checksums_are_not_trusted(
    tmp_path: Path, record: dict, monkeypatch
) -> None:
    payload = pickle.dumps(_ExecutableMetadata())
    directory = _write_source(tmp_path, record, payload=payload)
    (directory / "checksums.json").write_text(
        json.dumps({record["sample_id"]: hashlib.sha256(payload).hexdigest()})
    )
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    with pytest.raises(ValueError, match="checksum mismatch"):
        _dataset(directory, record)[0]


@pytest.mark.parametrize("valid_json", [False, True])
def test_json_precedence_never_falls_back_to_pickle(
    tmp_path: Path, record: dict, monkeypatch, valid_json: bool
) -> None:
    metadata = {"label": 1, "bands_order": record["metadata"]["bands_order"]}
    directory = _write_source(
        tmp_path,
        record,
        payload=pickle.dumps(_ExecutableMetadata()),
        json_metadata=json.dumps(metadata) if valid_json else "{",
    )
    monkeypatch.setattr(_metadata, "_MetadataUnpickler", _forbid_unpickling)
    if valid_json:
        assert _dataset(directory, record)[0]["label"].item() == 1
    else:
        with pytest.raises(json.JSONDecodeError):
            _dataset(directory, record)[0]


@pytest.mark.parametrize(
    "record",
    [r for r in RECORDS if r["layout"] == "hdf5"],
    ids=lambda r: f"{r['dataset_name']}-{r['sample_id']}",
)
def test_approved_hdf5_repacking_and_geography(tmp_path: Path, record: dict) -> None:
    source = _write_source(tmp_path / "source", record)
    output = tmp_path / "output" / source.name
    assert repack(source, output) == 1
    validate(source, output)
    copied = GeoBenchv1Sharded(output.parent, output.name, "train")[0]
    assert copied["label"].tolist() == record["metadata"]["label"]
    published_shard = _write_source(tmp_path / "published-shards", {**record, "layout": "sharded"})
    validate(source, published_shard)

    origin = _v1_origin(str(source / f"{record['sample_id']}.hdf5"))
    located = [
        entry
        for entry in record["metadata"].values()
        if isinstance(entry, dict) and entry.get("transform") and entry.get("crs")
    ]
    if located:
        entry = located[0]
        assert origin[:2] == pytest.approx((entry["transform"][2], entry["transform"][5]))
        assert CRS.from_user_input(origin[2]) == CRS.from_user_input(entry["crs"])
    else:
        assert origin == (None, None, "NOGEO")


@pytest.mark.parametrize("layout", ["hdf5", "sharded"])
def test_program_loads_published_pickle_metadata(tmp_path: Path, layout: str) -> None:
    records = [r for r in RECORDS if r["dataset_name"] == "m-eurosat" and r["layout"] == layout]
    record = records[0]
    subdir = "classification_v1.0" if layout == "hdf5" else "classification_v1.0_wds"
    source = _write_source(tmp_path / "data" / subdir, record)
    sid = record["sample_id"]
    (source / "default_partition.json").write_text(
        json.dumps({"train": [sid] * 8, "valid": [sid] * 2, "test": [sid] * 2})
    )
    output = tmp_path / "measurements.csv"
    arguments = [
        *classification_arguments(output),
        "dataset.num_workers=1",
        "eval.skip_linear=true",
    ]
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = pd.read_csv(output)
    assert rows["method"].tolist() == ["knn5"]
    assert rows["metric_value"].tolist() == [1.0]
    before = output.read_bytes()
    resumed = run_cli(*arguments, "resume=true", cwd=tmp_path)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert output.read_bytes() == before


@pytest.mark.parametrize("layout", ["hdf5", "sharded"])
def test_program_rejects_replaced_pickle_metadata(tmp_path: Path, layout: str) -> None:
    record = next(r for r in RECORDS if r["dataset_name"] == "m-eurosat" and r["layout"] == layout)
    subdir = "classification_v1.0" if layout == "hdf5" else "classification_v1.0_wds"
    _write_source(tmp_path / "data" / subdir, record, payload=pickle.dumps(_ExecutableMetadata()))
    output = tmp_path / "measurements.csv"
    completed = run_cli(*classification_arguments(output), cwd=tmp_path)
    assert completed.returncode != 0
    assert "metadata checksum mismatch" in completed.stderr
    assert not output.exists()
