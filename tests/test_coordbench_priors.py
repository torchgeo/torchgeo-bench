"""Tests for the separate label-informed CoordBench prior track."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from torchgeo_bench import coordbench
from torchgeo_bench.coordbench import (
    ClassFrequencyPrior,
    CoordBenchmark,
    CoordPriorConfig,
    GridPrior,
    KDEPrior,
    NearestNeighborPrior,
    UniformPrior,
    run_coordbench_priors,
)
from torchgeo_bench.coordbench.prior_run import RESUME_KEY_COLS, _fold_pairs
from torchgeo_bench.coordbench.probe import spatial_fold_ids


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


def _prior_cfg(tmp_path: Path, **overrides: Any) -> CoordPriorConfig:
    return CoordPriorConfig.model_validate(
        {
            "evaluation": {"split": "both", "folds": 3, **overrides},
            "output": {"file": str(tmp_path / "coordbench_priors.csv")},
        }
    )


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


def test_run_coordbench_priors_is_separate_and_classification_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.prior_run.load_benchmarks",
        lambda names: _synthetic_benchmarks(),
    )
    cfg = _prior_cfg(tmp_path)
    run_coordbench_priors(cfg)

    df = pd.read_csv(cfg.output.file)
    assert set(df.dataset) == {"synthetic-clf"}
    assert set(df.method) == {"prior"}
    assert set(df.model_name) == {"uniform", "frequency", "grid", "nearest", "kde"}
    assert set(df.split) == {"random", "spatial"}
    assert set(df.metric_name) == {"accuracy"}
    assert (df.n_test == 30).all()

    assert len(df) == 10
    assert (df.n_folds == 3).all()
    assert (df.feature_dim == 0).all()
    assert np.isfinite(df.metric_value).all()
    cfg.output.resume = True
    run_coordbench_priors(cfg)
    assert len(pd.read_csv(cfg.output.file)) == len(df)


def test_run_coordbench_priors_uses_official_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

    df = pd.read_csv(cfg.output.file)
    assert set(df.split) == {"official"}
    assert set(df.n_test) == {10}
    assert set(df.n_folds) == {1}


def test_aliases_remain_public() -> None:
    aliases = {
        "EmpiricalPrior": ClassFrequencyPrior,
        "DistancePrior": NearestNeighborPrior,
        "UniformBaseline": UniformPrior,
        "FrequencyBaseline": ClassFrequencyPrior,
        "GridBaseline": GridPrior,
        "NearestNeighborBaseline": NearestNeighborPrior,
        "KDEBaseline": KDEPrior,
    }
    for name, implementation in aliases.items():
        assert getattr(coordbench, name) is implementation


def test_probabilities_preserve_degree_coordinate_algorithms() -> None:
    lon = np.array([0.0, 0.0, 20.0])
    lat = np.zeros(3)
    labels = np.array(["a", "b", "b"])
    assert np.allclose(UniformPrior().fit(lon, lat, labels).predict_proba([90], [0]), [[0.5, 0.5]])
    assert np.allclose(
        ClassFrequencyPrior().fit(lon, lat, labels).predict_proba([90], [0]), [[1 / 3, 2 / 3]]
    )
    grid = GridPrior(smoothing=1.0).fit(lon, lat, labels)
    assert np.allclose(grid.predict_proba([20, 90], [0, 0]), [[1 / 3, 2 / 3], [1 / 3, 2 / 3]])
    nearest = NearestNeighborPrior(n_neighbors=20, weights="distance").fit(lon, lat, labels)
    assert np.allclose(nearest.predict_proba([0], [0]), [[0.5, 0.5]])
    kde = KDEPrior(bandwidth=1.0).fit(lon, lat, labels)
    assert kde.predict_proba([20], [0])[0, 1] > 0.99


def test_random_folds_are_torch_seeded_and_disjoint() -> None:
    pairs = _fold_pairs(12, 3, 7, None)
    expected = torch.randperm(12, generator=torch.Generator().manual_seed(7)).numpy()
    for index, (train, test) in enumerate(pairs):
        np.testing.assert_array_equal(test, expected[index::3])
        assert not set(train) & set(test)
        assert set(train) | set(test) == set(range(12))
    repeated = _fold_pairs(12, 3, 7, None)
    assert all(
        np.array_equal(first[1], second[1]) for first, second in zip(pairs, repeated, strict=True)
    )
    assert not np.array_equal(pairs[0][1], _fold_pairs(12, 3, 8, None)[0][1])


def test_spatial_folds_keep_cells_disjoint() -> None:
    lat = np.array([0, 1, 20, 21, 40, 41])
    lon = np.zeros(6)
    assignment = spatial_fold_ids(lat, lon, folds=3, cell_deg=10.0)
    pairs = _fold_pairs(6, 3, 0, assignment)
    assert len(pairs) == 3
    for train, test in pairs:
        assert not set(assignment[train]) & set(assignment[test])
        assert set(train) | set(test) == set(range(6))


def test_official_holdout_never_leaks_labels_and_counts_valid_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bench = CoordBenchmark(
        name="heldout",
        lon=np.arange(7, dtype=float),
        lat=np.zeros(7),
        tasks={"label": np.array([0, 0, np.nan, 1, 1, np.nan, 1])},
        task_type="classification",
        test_mask=np.array([False, False, False, True, True, True, True]),
    )
    bench.lon[-1] = np.nan
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.prior_run.load_benchmarks", lambda names: [bench]
    )
    fit = coordbench.SpatialPrior.fit
    seen = []

    def fit_training(self, lon, lat, labels):
        np.testing.assert_array_equal(lon, [0, 1])
        np.testing.assert_array_equal(labels, [0, 0])
        seen.append(type(self))
        return fit(self, lon, lat, labels)

    monkeypatch.setattr(coordbench.SpatialPrior, "fit", fit_training)
    cfg = _prior_cfg(tmp_path)
    run_coordbench_priors(cfg)
    df = pd.read_csv(cfg.output.file)
    assert len(seen) == len(df) == 5
    assert (df.metric_value == 0).all()
    assert (df.n_test == 2).all()
    assert (df.n_samples == 7).all()
    assert (df.n_folds == 1).all()
    assert (df.ci_lower == df.ci_upper).all()


@pytest.mark.parametrize("empty_side", ["train", "test", "invalid", "no_samples", "one_cell"])
def test_empty_splits_warn_without_nan_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    empty_side: str,
) -> None:
    bench = _synthetic_benchmarks()[0]
    if empty_side in {"train", "test", "invalid"}:
        bench.test_mask = np.full(30, empty_side == "train")
        if empty_side == "invalid":
            bench.tasks = {"label": np.full(30, np.nan)}
    elif empty_side == "no_samples":
        bench.lon = bench.lat = np.array([])
        bench.tasks = {"label": np.array([])}
    else:
        bench.lon = bench.lat = np.zeros(30)
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.prior_run.load_benchmarks", lambda names: [bench]
    )
    cfg = _prior_cfg(tmp_path, split="spatial")
    run_coordbench_priors(cfg)
    assert not Path(cfg.output.file).exists()
    assert "no non-empty valid training/test split" in caplog.text


def test_resume_uses_prior_identity_and_evaluates_missing_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "torchgeo_bench.coordbench.prior_run.load_benchmarks", lambda names: _synthetic_benchmarks()
    )
    cfg = _prior_cfg(tmp_path, methods=["uniform"], split="random")
    run_coordbench_priors(cfg)
    original = pd.read_csv(cfg.output.file)
    cfg.evaluation.methods = ["uniform", "frequency"]
    cfg.output.resume = True
    run_coordbench_priors(cfg)
    results = pd.read_csv(cfg.output.file)
    assert len(results) == 2
    pd.testing.assert_frame_equal(results.iloc[:1], original)
    assert not results.duplicated(list(RESUME_KEY_COLS)).any()
