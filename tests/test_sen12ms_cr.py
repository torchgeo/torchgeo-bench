import pickle
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from torchgeo.datasets.errors import DatasetNotFoundError


def _write_pickle(path: Path, obj: object) -> None:
    path.write_bytes(pickle.dumps(obj))


def _make_samples(n: int, *, include_cloudy: bool = False, offset: int = 0) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for i in range(n):
        sid = str(offset + i)
        sample: dict[str, object] = {
            "id": sid,
            "s2": f"ROIs1158_spring/s2_1/{sid}.tif",
        }
        if include_cloudy:
            sample["s2_cloudy"] = f"ROIs1158_spring/s2_cloudy_1/{sid}.tif"
        samples.append(sample)
    return samples


def _make_fake_root(
    tmp_path: Path,
    *,
    n_train: int = 4,
    n_val: int = 2,
    n_test: int = 5,
    n_cloudy: int = 6,
) -> Path:
    root = tmp_path / "sen12ms_cr"
    root.mkdir(parents=True)
    (root / "ROIs1158_spring").mkdir()

    train = _make_samples(n_train, offset=0)
    val = _make_samples(n_val, offset=n_train)
    test = _make_samples(n_test, offset=n_train + n_val)
    cloudy = _make_samples(n_cloudy, include_cloudy=True, offset=n_train + n_val + n_test)

    _write_pickle(root / "train_list.pkl", train)
    _write_pickle(root / "val_list.pkl", val)
    _write_pickle(root / "test_list.pkl", test)
    _write_pickle(root / "test_list_cloudy.pkl", cloudy)

    all_ids = [str(s["id"]) for s in [*train, *val, *test, *cloudy]]
    labels = {sid: np.eye(10, dtype=np.float32)[int(sid) % 10] for sid in all_ids}
    _write_pickle(root / "IGBP_probability_labels.pkl", labels)

    coverage: dict[str, float] = {}
    for idx, sample in enumerate(cloudy):
        sid = str(sample["id"])
        if idx == len(cloudy) - 1:
            coverage[sid] = 100.0
        else:
            coverage[sid] = float(idx * (100.0 / len(cloudy)))
    _write_pickle(root / "cloud_coverage.pkl", coverage)
    return root


def test_class_attributes():
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS

    assert SEN12MS.task == "classification"
    assert SEN12MS.num_classes == 10
    assert SEN12MS.multilabel is False
    assert SEN12MS.rgb_bands == ["red", "green", "blue"]


@pytest.mark.parametrize(
    ("cls_name", "expected"),
    [
        ("SEN12MS", None),
        ("SEN12MSCRC1", (0.0, 20.0)),
        ("SEN12MSCRC3", (40.0, 60.0)),
        ("SEN12MSCRC5", (80.0, 100.0)),
    ],
)
def test_cloud_bin_values(cls_name: str, expected: tuple[float, float] | None):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS, SEN12MSCRC1, SEN12MSCRC3, SEN12MSCRC5

    by_name = {
        "SEN12MS": SEN12MS,
        "SEN12MSCRC1": SEN12MSCRC1,
        "SEN12MSCRC3": SEN12MSCRC3,
        "SEN12MSCRC5": SEN12MSCRC5,
    }
    assert by_name[cls_name]._cloud_bin == expected


def test_prior_results_alias():
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS, SEN12MSCRC1

    assert SEN12MS.prior_results_alias is None
    assert SEN12MSCRC1.prior_results_alias == "sen12ms"


def test_split_sizes_clean(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS

    root = _make_fake_root(tmp_path, n_train=3, n_val=2, n_test=4, n_cloudy=7)
    with patch.object(SEN12MS, "data_root", return_value=root):
        bench = SEN12MS()
    assert bench.split_sizes == {"train": 3, "val": 2, "test": 4}


def test_split_sizes_cloud_bin(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MSCRC1, SEN12MSCRC5

    root = _make_fake_root(tmp_path, n_cloudy=6)
    cloudy = pickle.loads((root / "test_list_cloudy.pkl").read_bytes())
    coverage = {
        str(cloudy[0]["id"]): 0.0,
        str(cloudy[1]["id"]): 10.0,
        str(cloudy[2]["id"]): 30.0,
        str(cloudy[3]["id"]): 50.0,
        str(cloudy[4]["id"]): 80.0,
        str(cloudy[5]["id"]): 100.0,
    }
    _write_pickle(root / "cloud_coverage.pkl", coverage)

    with patch.object(SEN12MSCRC1, "data_root", return_value=root):
        c1 = SEN12MSCRC1()
    with patch.object(SEN12MSCRC5, "data_root", return_value=root):
        c5 = SEN12MSCRC5()

    assert c1.split_sizes["test"] == 2  # 0 and 10 in [0, 20)
    assert c5.split_sizes["test"] == 2  # 80 and 100 in [80, 100]


def test_get_dataset_returns_correct_length(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS

    root = _make_fake_root(tmp_path, n_train=5, n_val=4, n_test=3, n_cloudy=8)
    with patch.object(SEN12MS, "data_root", return_value=root):
        bench = SEN12MS()
    assert len(bench.get_dataset("train")) == 5
    assert len(bench.get_dataset("val")) == 4
    assert len(bench.get_dataset("test")) == 3


def test_get_dataset_resolves_relative_paths_only(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS

    root = _make_fake_root(tmp_path)
    train = pickle.loads((root / "train_list.pkl").read_bytes())
    train[0]["s2"] = "/tmp/abs.tif"
    _write_pickle(root / "train_list.pkl", train)

    with patch.object(SEN12MS, "data_root", return_value=root):
        bench = SEN12MS()
        with pytest.raises(ValueError, match="absolute path"):
            bench.get_dataset("train")


def test_cloud_variant_uses_cloudy_paths_for_test_split(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MSCRC3

    root = _make_fake_root(tmp_path, n_cloudy=9)
    with patch.object(SEN12MSCRC3, "data_root", return_value=root):
        bench = SEN12MSCRC3()
    view = bench.get_dataset("test")
    assert view.items  # type: ignore[attr-defined]
    assert all("s2_cloudy" in str(path) for path, _ in view.items)  # type: ignore[attr-defined]


def test_igbp_argmax_applied(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS

    root = _make_fake_root(tmp_path, n_train=2, n_val=1, n_test=1, n_cloudy=1)
    labels = pickle.loads((root / "IGBP_probability_labels.pkl").read_bytes())
    labels["0"] = np.array([0.1, 0.7, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    _write_pickle(root / "IGBP_probability_labels.pkl", labels)

    with patch.object(SEN12MS, "data_root", return_value=root):
        bench = SEN12MS()
    train_view = bench.get_dataset("train")
    first_label = train_view.items[0][1]  # type: ignore[attr-defined]
    assert first_label == 1


def test_check_data_present_raises_clean_missing(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS

    root = tmp_path / "sen12ms_cr"
    root.mkdir()
    with (
        patch.object(SEN12MS, "data_root", return_value=root),
        pytest.raises(DatasetNotFoundError),
    ):
        SEN12MS()


def test_check_data_present_raises_cloud_missing(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MSCRC2

    root = _make_fake_root(tmp_path)
    (root / "cloud_coverage.pkl").unlink()
    with (
        patch.object(SEN12MSCRC2, "data_root", return_value=root),
        pytest.raises(DatasetNotFoundError),
    ):
        SEN12MSCRC2()


def test_missing_label_or_coverage_entry_raises(tmp_path: Path):
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MS, SEN12MSCRC2

    root_missing_label = _make_fake_root(tmp_path / "a")
    labels = pickle.loads((root_missing_label / "IGBP_probability_labels.pkl").read_bytes())
    labels.pop("0")
    _write_pickle(root_missing_label / "IGBP_probability_labels.pkl", labels)
    with patch.object(SEN12MS, "data_root", return_value=root_missing_label):
        bench = SEN12MS()
        with pytest.raises(ValueError, match="Missing label.*sample_id=0"):
            bench.get_dataset("train")

    root_missing_coverage = _make_fake_root(tmp_path / "b")
    coverage = pickle.loads((root_missing_coverage / "cloud_coverage.pkl").read_bytes())
    first_cloudy_id = str(pickle.loads((root_missing_coverage / "test_list_cloudy.pkl").read_bytes())[0]["id"])
    coverage.pop(first_cloudy_id)
    _write_pickle(root_missing_coverage / "cloud_coverage.pkl", coverage)
    with (
        patch.object(SEN12MSCRC2, "data_root", return_value=root_missing_coverage),
        pytest.raises(ValueError, match=f"Missing cloud coverage.*sample_id={first_cloudy_id}"),
    ):
        SEN12MSCRC2()


def test_registered_names():
    from torchgeo_bench.datasets import get_bench_dataset_class, list_datasets
    from torchgeo_bench.datasets.sen12ms_cr import SEN12MSCRC3

    for name in [
        "sen12ms",
        "sen12ms_cr_c1",
        "sen12ms_cr_c2",
        "sen12ms_cr_c3",
        "sen12ms_cr_c4",
        "sen12ms_cr_c5",
    ]:
        cls = get_bench_dataset_class(name)
        assert cls.name == name

    assert get_bench_dataset_class("sen12ms_cr_c3") is SEN12MSCRC3
    assert {"sen12ms", "sen12ms_cr_c1", "sen12ms_cr_c2", "sen12ms_cr_c3", "sen12ms_cr_c4", "sen12ms_cr_c5"}.issubset(set(list_datasets()))


def test_public_import():
    from torchgeo_bench.datasets import SEN12MS, SEN12MSCRC1, SEN12MSCRC5

    assert SEN12MS.name == "sen12ms"
    assert SEN12MSCRC1.name == "sen12ms_cr_c1"
    assert SEN12MSCRC5.name == "sen12ms_cr_c5"
