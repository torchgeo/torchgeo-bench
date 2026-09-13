"""Tests for the CoordBench location-encoder track (network-free)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from tests.support.numerical import isolated_torch_rng as isolated_torch_rng
from torchgeo_bench.coordbench import (
    CoordBenchmark,
    SinCosLocationEncoder,
    knn_probe_score,
    linear_probe_score,
    load_benchmarks,
    run_coordbench,
    spatial_fold_ids,
)
from torchgeo_bench.coordbench import datasets as cb_datasets

pytestmark = pytest.mark.usefixtures("isolated_torch_rng")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(-180, 180, size=400)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=400)))
    return lon, lat


@pytest.mark.parametrize("batch_size", [1, 137, 8192])
def test_sincos_encode_preserves_coordinate_order_across_batches(
    points: tuple[np.ndarray, np.ndarray], batch_size: int
) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu", batch_size=batch_size).encode(lon, lat)
    assert feats.dtype == np.float32
    lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
    expected = np.column_stack([np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)])
    np.testing.assert_allclose(feats, expected, rtol=1e-6, atol=1e-7)


def test_documented_fourier_encoder_example(points: tuple[np.ndarray, np.ndarray]) -> None:
    from examples.coordbench_location_encoder import FourierLocationEncoder

    lon, lat = points
    feats = FourierLocationEncoder(num_frequencies=4).encode(lon, lat)
    assert feats.shape == (len(lon), 16)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


def test_linear_probe_regression_recovers_smooth_target(
    points: tuple[np.ndarray, np.ndarray],
) -> None:
    # Ridge regression should recover a target built directly from the features.
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = 3.0 * feats[:, 0] - 2.0 * feats[:, 3] + 0.5  # 3*sin(lat) - 2*cos(lon) + b
    score, fold_scores = linear_probe_score(feats, labels, "regression", device="cpu")
    assert len(fold_scores) == 5
    assert score > 0.98


def test_linear_and_knn_classification(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = (lat > 0).astype(np.int64)  # The sign of sin(latitude) separates the hemispheres.
    lin, _ = linear_probe_score(feats, labels, "classification", device="cpu")
    knn, folds = knn_probe_score(feats, labels, device="cpu", k=5)
    assert lin > 0.9
    assert knn > 0.9
    assert len(folds) == 5


def test_spatial_fold_ids_group_by_cell(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    fa = spatial_fold_ids(lat, lon, folds=5, cell_deg=10.0, seed=0)
    assert fa.shape == (len(lon),)
    assert set(np.unique(fa)).issubset(set(range(5)))
    cell = np.floor(lat / 10.0).astype(int) * 100003 + np.floor(lon / 10.0).astype(int)
    for c in np.unique(cell):
        assert len(np.unique(fa[cell == c])) == 1


def test_test_mask_split_single_fold(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    labels = 3.0 * feats[:, 0] + 0.5
    mask = np.zeros(len(lon), dtype=bool)
    mask[::4] = True  # 25% held out
    _, fold_scores = linear_probe_score(feats, labels, "regression", device="cpu", test_mask=mask)
    assert len(fold_scores) == 1  # An official split has one held-out score, not a CV average.


def _synthetic_benchmarks() -> list[CoordBenchmark]:
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 180, size=300)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=300)))
    reg = CoordBenchmark(
        name="synthetic-reg",
        lat=lat,
        lon=lon,
        tasks={"target": np.sin(np.deg2rad(lat)) + 0.1 * rng.standard_normal(300)},
        task_type="regression",
    )
    clf = CoordBenchmark(
        name="synthetic-clf",
        lat=lat,
        lon=lon,
        tasks={"label": (lat > 0).astype(np.int64)},
        task_type="classification",
    )
    return [reg, clf]


def _coord_cfg(tmp_path: Path, **coord_overrides: object) -> DictConfig:
    coord = {
        "output": str(tmp_path / "coord.csv"),
        "names": "all",
        "methods": ["knn", "linear"],
        "split": "random",
        "folds": 5,
        "cell_deg": 10.0,
        "knn_k": 5,
    }
    coord.update(coord_overrides)
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "resume": False,
            "mode": "coord",
            "model": {
                "_target_": "torchgeo_bench.coordbench.models.SinCosLocationEncoder",
                "name": "sincos",
            },
            "coord": coord,
        }
    )


def test_run_coordbench_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path, split="both")
    run_coordbench(cfg)

    df = pd.read_csv(cfg.coord.output)
    assert {"dataset", "task", "method", "split", "metric_name", "metric_value"} <= set(df.columns)
    reg = df[df.dataset == "synthetic-reg"]
    clf = df[df.dataset == "synthetic-clf"]
    assert set(reg.method) == {"linear"}
    assert set(reg.metric_name) == {"r2"}
    assert set(clf.method) == {"linear", "knn5"}
    assert set(clf.metric_name) == {"accuracy"}
    # Without an official split, both cross-validation schemes should run.
    assert {"random", "spatial"} <= set(df.split)
    assert (df.metric_value.abs() <= 1.5).all()


def test_run_coordbench_resume_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path)
    run_coordbench(cfg)
    before = Path(cfg.coord.output).read_bytes()

    def unexpected_probe(*args: object, **kwargs: object) -> None:
        pytest.fail("Completed coordinate probes must not be recomputed")

    cfg.resume = True
    monkeypatch.setattr("torchgeo_bench.coordbench.run.linear_probe_score", unexpected_probe)
    monkeypatch.setattr("torchgeo_bench.coordbench.run.knn_probe_score", unexpected_probe)
    run_coordbench(cfg)
    assert Path(cfg.coord.output).read_bytes() == before


def test_run_coordbench_reports_official_test_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _synthetic_benchmarks()[0]
    test_mask = np.zeros(len(bench.lat), dtype=bool)
    test_mask[::3] = True
    bench.test_mask = test_mask
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda names: [bench])

    cfg = _coord_cfg(tmp_path, split="both")
    run_coordbench(cfg)

    df = pd.read_csv(cfg.coord.output)
    assert set(df.split) == {"official"}
    assert set(df.n_test) == {int(test_mask.sum())}


def test_load_benchmarks_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    # Use local tables instead of downloading Parquet files.
    tables = {
        "country": pd.DataFrame(
            {"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0], "country": [1, 2, 1]}
        ),
        "worldclim_bio": pd.DataFrame(
            {"lon": [0.0, 1.0], "lat": [0.0, 1.0], "bio1": [10.0, 20.0], "bio12": [1.0, 2.0]}
        ),
    }
    monkeypatch.setattr(cb_datasets, "load_config", lambda cfg: tables[cfg])

    only_country = load_benchmarks("country")
    assert [b.name for b in only_country] == ["country"]

    wc = load_benchmarks("worldclim")
    assert {b.name for b in wc} == {"worldclim-bio1", "worldclim-bio12"}

    single = load_benchmarks("worldclim-bio1")
    assert [b.name for b in single] == ["worldclim-bio1"]

    np.testing.assert_array_equal(single[0].tasks["bio1"], [10.0, 20.0])


def test_mind_load_roundtrip(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    from torchgeo_bench.coordbench.mind import ReSIRENLocationEncoder, load_mind

    model = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2).eval()
    path = tmp_path / "m.safetensors"
    save_file(model.state_dict(), str(path))

    loaded = load_mind(str(path))
    assert loaded.embed_dim == 16
    assert len(loaded.blocks) == 2
    assert not loaded.use_year  # A two-input checkpoint has coordinates but no year.

    latlon = torch.tensor([[37.77, -122.42], [51.51, -0.13]], dtype=torch.float32)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(latlon, return_features=True), model(latlon, return_features=True)
        )
        torch.testing.assert_close(loaded(latlon), model(latlon))


def test_mind_encoder_dim_slice(monkeypatch: pytest.MonkeyPatch) -> None:
    from torchgeo_bench.coordbench import mind as mind_mod
    from torchgeo_bench.coordbench.models import MINDLocationEncoder

    model = mind_mod.ReSIRENLocationEncoder(embed_dim=32, out_dim=32, depth=2).eval()
    monkeypatch.setattr(mind_mod, "load_mind", lambda path, device="cpu": model)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "dummy")

    enc = MINDLocationEncoder(dim=8, feature="pooled", device="cpu")
    out = enc.encode(np.array([1.0, 2.0]), np.array([37.0, 51.0]))
    assert out.shape == (2, 8)  # MIND's training permits using a prefix of the 32 features.
    assert out.dtype == np.float32
    with torch.no_grad():
        expected = model(torch.tensor([[37.0, 1.0], [51.0, 2.0]]), return_features=True)
    np.testing.assert_allclose(out, expected[:, :8].numpy())


def test_family_index_matches_loaders() -> None:
    # Static family indexes allow listing and selection without downloads.
    assert set(cb_datasets.FAMILY_BENCHMARKS) == set(cb_datasets.FAMILY_LOADERS)
    all_names = cb_datasets.list_benchmarks()
    assert len(all_names) == len(set(all_names))
    assert set(all_names) == {
        name for family_names in cb_datasets.FAMILY_BENCHMARKS.values() for name in family_names
    }
