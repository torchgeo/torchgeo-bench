"""Tests for the separate label-informed CoordBench prior track."""

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from torchgeo_bench.coordbench import (
    ClassFrequencyPrior,
    CoordBenchmark,
    GridPrior,
    KDEPrior,
    NearestNeighborPrior,
    UniformPrior,
    run_coordbench_priors,
)


def test_spatial_priors_return_probabilities() -> None:
    lon = np.array([-100.0, -99.0, 10.0, 11.0])
    lat = np.array([40.0, 41.0, 10.0, 11.0])
    labels = np.array(["a", "a", "b", "b"])
    for prior in (
        UniformPrior(),
        ClassFrequencyPrior(),
        GridPrior(),
        NearestNeighborPrior(),
        NearestNeighborPrior(weights="distance"),
        KDEPrior(),
    ):
        probabilities = prior.fit(lon, lat, labels).predict_proba(lon, lat)
        assert probabilities.shape == (4, 2)
        assert np.isfinite(probabilities).all()
        assert np.allclose(probabilities.sum(axis=1), 1.0)


def _prior_cfg(tmp_path, **overrides) -> OmegaConf:
    coord_prior = {
        "output": str(tmp_path / "coordbench_priors.csv"),
        "names": "all",
        "methods": ["uniform", "frequency", "grid", "nearest", "kde"],
        "split": "both",
        "folds": 3,
        "cell_deg": 10.0,
        "grid_cell_size": 10.0,
        "smoothing": 0.0,
        "nearest_k": 5,
        "nearest_weights": "uniform",
        "kde_bandwidth": 10.0,
    }
    coord_prior.update(overrides)
    return OmegaConf.create({"seed": 0, "resume": False, "coord_prior": coord_prior})


def _synthetic_benchmarks() -> list[CoordBenchmark]:
    lon = np.linspace(-120.0, -80.0, 30)
    lat = np.linspace(20.0, 50.0, 30)
    return [
        CoordBenchmark(
            name="synthetic-clf",
            lat=lat,
            lon=lon,
            tasks={"label": (lat > 35.0).astype(np.int64)},
            task_type="classification",
        ),
        CoordBenchmark(
            name="synthetic-reg",
            lat=lat,
            lon=lon,
            tasks={"target": lat},
            task_type="regression",
        ),
    ]


def test_run_coordbench_priors_is_separate_and_classification_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.prior_run.load_benchmarks",
        lambda names: _synthetic_benchmarks(),
    )
    cfg = _prior_cfg(tmp_path)
    run_coordbench_priors(cfg)

    df = pd.read_csv(cfg.coord_prior.output)
    assert set(df.dataset) == {"synthetic-clf"}
    assert set(df.method) == {"prior"}
    assert set(df.model_name) == {"uniform", "frequency", "grid", "nearest", "kde"}
    assert set(df.split) == {"random", "spatial"}
    assert set(df.metric_name) == {"accuracy"}
    assert (df.n_test == 30).all()

    cfg.resume = True
    run_coordbench_priors(cfg)
    assert len(pd.read_csv(cfg.coord_prior.output)) == len(df)


def test_run_coordbench_priors_uses_official_holdout(tmp_path, monkeypatch) -> None:
    bench = _synthetic_benchmarks()[0]
    test_mask = np.zeros(len(bench.lat), dtype=bool)
    test_mask[::3] = True
    bench.test_mask = test_mask
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.prior_run.load_benchmarks",
        lambda names: [bench],
    )
    cfg = _prior_cfg(tmp_path, methods=["frequency"], split="both")
    run_coordbench_priors(cfg)

    df = pd.read_csv(cfg.coord_prior.output)
    assert set(df.split) == {"official"}
    assert set(df.n_test) == {10}
    assert set(df.n_folds) == {1}
