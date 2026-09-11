"""Tests for the CoordBench location-encoder track (network-free)."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

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
from torchgeo_bench.coordbench.config import CoordConfig
from torchgeo_bench.coordbench.run import _instantiate_encoder
from torchgeo_bench.presets import ModelPreset


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def points(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    lon = rng.uniform(-180, 180, size=400)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=400)))
    return lon, lat


def test_sincos_encode_shape(points: tuple[np.ndarray, np.ndarray]) -> None:
    lon, lat = points
    feats = SinCosLocationEncoder(device="cpu").encode(lon, lat)
    assert feats.shape == (len(lon), 4)
    assert feats.dtype == np.float32
    assert np.isfinite(feats).all()


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


def _coord_cfg(tmp_path, **coord_overrides) -> CoordConfig:
    coord = {
        "methods": ["knn", "linear"],
        "split": "random",
        "folds": 5,
        "cell_deg": 10.0,
        "knn_k": 5,
    }
    coord.update(coord_overrides)
    return CoordConfig.model_validate(
        {
            "runtime": {"seed": 0, "device": "cpu"},
            "output": {"file": str(tmp_path / "coord.csv"), "resume": False},
            "model": {"name": "sincos"},
            "evaluation": coord,
        }
    )


def test_run_coordbench_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path, split="both")
    run_coordbench(cfg)

    df = pd.read_csv(cfg.output.file)
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


def test_run_coordbench_resume_skips(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    cfg = _coord_cfg(tmp_path)
    run_coordbench(cfg)
    n_first = len(pd.read_csv(cfg.output.file))

    cfg.output.resume = True
    run_coordbench(cfg)
    assert len(pd.read_csv(cfg.output.file)) == n_first


def test_run_coordbench_reports_official_test_count(tmp_path, monkeypatch) -> None:
    bench = _synthetic_benchmarks()[0]
    test_mask = np.zeros(len(bench.lat), dtype=bool)
    test_mask[::3] = True
    bench.test_mask = test_mask
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda names: [bench])

    cfg = _coord_cfg(tmp_path, split="both")
    run_coordbench(cfg)

    df = pd.read_csv(cfg.output.file)
    assert set(df.split) == {"official"}
    assert set(df.n_test) == {int(test_mask.sum())}


def test_load_benchmarks_selection(monkeypatch) -> None:
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

    assert "pdfm" in cb_datasets.list_families()


def test_mind_load_roundtrip(tmp_path) -> None:
    from safetensors.torch import save_file

    from torchgeo_bench.coordbench.mind import ReSIRENLocationEncoder, load_mind

    torch.manual_seed(0)
    model = ReSIRENLocationEncoder(embed_dim=16, out_dim=8, depth=2)
    path = tmp_path / "m.safetensors"
    save_file(model.state_dict(), str(path))

    loaded = load_mind(str(path))
    assert loaded.embed_dim == 16
    assert len(loaded.blocks) == 2
    assert not loaded.use_year  # A two-input checkpoint has coordinates but no year.

    latlon = torch.tensor([[37.77, -122.42], [51.51, -0.13]], dtype=torch.float32)
    assert loaded(latlon, return_features=True).shape == (2, 16)
    assert loaded(latlon).shape == (2, 8)


def test_mind_encoder_dim_slice(monkeypatch) -> None:
    from torchgeo_bench.coordbench import mind as mind_mod
    from torchgeo_bench.coordbench.models import MINDLocationEncoder

    torch.manual_seed(0)
    model = mind_mod.ReSIRENLocationEncoder(embed_dim=32, out_dim=32, depth=2).eval()
    monkeypatch.setattr(mind_mod, "load_mind", lambda path, device="cpu": model)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *a, **k: "dummy")

    enc = MINDLocationEncoder(dim=8, feature="pooled", device="cpu")
    out = enc.encode(np.array([1.0, 2.0]), np.array([37.0, 51.0]))
    assert out.shape == (2, 8)  # MIND's training permits using a prefix of the 32 features.
    assert out.dtype == np.float32


def test_family_index_matches_loaders() -> None:
    # Static family indexes allow listing and selection without downloads.
    assert set(cb_datasets.FAMILY_BENCHMARKS) == set(cb_datasets.FAMILY_LOADERS)
    all_names = cb_datasets.list_benchmarks()
    assert len(all_names) == len(set(all_names))
    assert "pdfm-conus27" in all_names
    assert sum(n.startswith("dm-") for n in all_names) == 15


@pytest.mark.parametrize("methods", [["linear"], ["knn"], ["knn", "linear"]])
def test_runtime_method_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, methods: list[str]
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    calls = []

    def knn(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        calls.append(("knn", kwargs))
        return 0.8, [0.7, 0.9]

    def linear(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        calls.append(("linear", kwargs))
        return 0.6, [0.5, 0.7]

    monkeypatch.setattr("torchgeo_bench.coordbench.run.knn_probe_score", knn)
    monkeypatch.setattr("torchgeo_bench.coordbench.run.linear_probe_score", linear)
    config = _coord_cfg(tmp_path, methods=methods, knn_k=7, knn_device="cpu")
    run_coordbench(config)
    rows = pd.read_csv(config.output.file)
    assert {kind for kind, _ in calls} == set(methods)
    assert set(rows.method) == {"knn7" if method == "knn" else "linear" for method in methods}
    assert all(options["device"] == "cpu" for _, options in calls)
    assert set(rows.model_target) == {"torchgeo_bench.coordbench.models.SinCosLocationEncoder"}
    assert set(rows.model_name) == {"sincos"}
    assert list(rows.columns) == [
        "dataset",
        "task",
        "task_type",
        "method",
        "split",
        "metric_name",
        "metric_value",
        "ci_lower",
        "ci_upper",
        "n_folds",
        "cell_deg",
        "feature_dim",
        "n_samples",
        "n_test",
        "seed",
        "model_name",
        "model_target",
    ]


def test_resume_skips_encoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    config = _coord_cfg(tmp_path)
    run_coordbench(config)
    before = Path(config.output.file).read_bytes()

    def unexpected_encode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Completed benchmarks must not be encoded again")

    monkeypatch.setattr(SinCosLocationEncoder, "encode", unexpected_encode)
    config.output.resume = True
    run_coordbench(config)
    assert Path(config.output.file).read_bytes() == before


def test_partial_results_survive_failure_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = _synthetic_benchmarks()[0]
    bench.tasks["second"] = bench.tasks["target"].copy()
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda names: [bench])
    calls = []

    def failing_linear(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        calls.append("linear")
        if len(calls) == 2:
            raise RuntimeError("second task failed")
        return 0.8, [0.8]

    monkeypatch.setattr("torchgeo_bench.coordbench.run.linear_probe_score", failing_linear)
    config = _coord_cfg(tmp_path, methods=["linear"])
    with pytest.raises(RuntimeError, match="second task failed"):
        run_coordbench(config)
    assert pd.read_csv(config.output.file).task.tolist() == ["target"]
    config.output.resume = True
    run_coordbench(config)
    assert pd.read_csv(config.output.file).task.tolist() == ["target", "second"]
    assert len(calls) == 3


def test_knn_only_skips_regression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks",
        lambda names: [_synthetic_benchmarks()[0]],
    )

    def unexpected_encode(*args: Any, **kwargs: Any) -> None:
        pytest.fail("KNN-only regression must not encode or probe")

    monkeypatch.setattr(SinCosLocationEncoder, "encode", unexpected_encode)
    config = _coord_cfg(tmp_path, methods=["knn"])
    run_coordbench(config)
    assert not Path(config.output.file).exists()


class FixtureEncoder(SinCosLocationEncoder):
    def __init__(self, payload: dict[str, Any], device: str) -> None:
        super().__init__(device=device)
        self.payload = payload


def test_custom_encoder_preserves_ordinary_nested_kwargs() -> None:
    payload = {"_target_": "not_imported.Missing", "options": {"enabled": True}}
    preset = ModelPreset(
        name="my-encoder",
        target=f"{__name__}.FixtureEncoder",
        kwargs={"payload": payload},
    )
    encoder = _instantiate_encoder(preset, "cpu")
    assert isinstance(encoder, FixtureEncoder)
    assert encoder.payload == payload
    assert encoder.device == "cpu"


def test_rejects_non_location_custom_target() -> None:
    preset = ModelPreset(name="not-an-encoder", target="builtins.dict")
    with pytest.raises(TypeError, match="LocationEncoder"):
        _instantiate_encoder(preset, "cpu")


def test_auto_device_resolves_on_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.run.load_benchmarks",
        lambda names: [_synthetic_benchmarks()[0]],
    )
    devices = []

    def linear(*args: Any, **kwargs: Any) -> tuple[float, list[float]]:
        devices.append(kwargs["device"])
        return 0.5, [0.5]

    monkeypatch.setattr("torchgeo_bench.coordbench.run.linear_probe_score", linear)
    config = _coord_cfg(tmp_path)
    config.runtime.device = "auto"
    run_coordbench(config)
    assert devices == ["cpu"]
    assert config.runtime.device == "auto"


def test_runtime_seeds_encoder_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda names: [])
    random_values = []

    def build(preset: ModelPreset, *, device: str) -> SinCosLocationEncoder:
        random_values.append(torch.rand(4))
        return SinCosLocationEncoder(device=device)

    monkeypatch.setattr("torchgeo_bench.coordbench.run.build_model", build)
    config = _coord_cfg(tmp_path)
    run_coordbench(config)
    run_coordbench(config)
    assert torch.equal(random_values[0], random_values[1])
