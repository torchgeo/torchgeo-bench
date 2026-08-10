"""Unit tests for utility helpers in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import torchgeo_bench
from torchgeo_bench.main import (
    _RESUME_KEY_COLS,
    EvaluationResult,
    _build_seg_probe_and_solver,
    _canonical_key_cell,
    _completed_run_keys,
    _expand_dataset_list,
    _filter_completed_metric_rows,
    _measure_cpu_throughput,
    _normalize_bands_value,
    _normalize_resume_fraction_columns,
    _plan_dataset_run,
    _row_key,
    _run_linear_fractions,
    evaluate_profile,
)
from torchgeo_bench.utils import stratified_subsample_indices


def _cached(n: int, d: int = 8, n_classes: int = 3, seed: int = 0):
    """Synthetic cached embedding/label matrices with every class present."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    # Guarantee each class appears at least once, then fill randomly.
    y = np.concatenate(
        [np.arange(n_classes), rng.integers(0, n_classes, size=n - n_classes)]
    ).astype(np.int64)
    return x, y


def _fraction_common_meta() -> dict:
    return {
        "dataset": "m-eurosat",
        "seed": 0,
        "model": "mock.Model",
        "name": "mock",
        "normalization": "identity",
        "image_size": 8,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "c_range_start": -2,
        "c_range_stop": 2,
        "c_range_num": 3,
        "merge_val": True,
        "bootstrap": 10,
    }


def _fake_logistic(*args, **kwargs):
    cal = {"ece": 0.1, "rms_ce": 0.1, "mce": 0.2}
    cal_ts = {"ece_ts": 0.05, "rms_ce_ts": 0.05, "mce_ts": 0.1, "temperature": 1.2}
    return 0.9, 0.85, 0.95, 1.0, cal, cal_ts


def _call_run_linear_fractions(pairs, x_train, y_train, x_val, y_val, x_test, y_test):
    return _run_linear_fractions(
        pairs=pairs,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        c_values=[0.1, 1.0, 10.0],
        common_meta=_fraction_common_meta(),
        metric_name="accuracy",
        feature_dim=x_train.shape[1],
        seed=0,
        n_bootstrap=10,
        device="cpu",
        verbose=False,
        calibration_n_bins=15,
        temp_scale=True,
    )


class _FakeClsDataset:
    """Minimal single-label classification dataset class for planning tests."""

    task = "classification"
    multilabel = False
    num_classes = 5


def _plan_config(
    label_fractions, n_subsample_repeats, *, seed=0, resume=False, skip_linear=False
) -> OmegaConf:
    return OmegaConf.create(
        {
            "seed": seed,
            "resume": resume,
            "model": {"_target_": "mock.Model", "name": "mock"},
            "eval": {
                "skip_linear": skip_linear,
                "knn_k": 5,
                "label_fractions": list(label_fractions),
                "n_subsample_repeats": n_subsample_repeats,
                "segmentation": {"head_type": "fpn"},
            },
        }
    )


def _plan_config_tuple(seed=0) -> tuple[str, ...]:
    """Config tuple as assembled in ``main()`` (base 1.0/seed cells appended)."""
    return tuple(
        _canonical_key_cell(v) for v in ("identity", 8, "bilinear", "default", "rgb", 1.0, seed)
    )


def _plan_linear_key(ds, fraction, subsample_seed, *, cfg_seed=0) -> tuple[str, ...]:
    ct = _plan_config_tuple(cfg_seed)
    base = ("mock.Model", "mock", *ct[:-2])
    return (
        ds,
        "linear",
        *base,
        _canonical_key_cell(fraction),
        _canonical_key_cell(subsample_seed),
    )


def _plan_knn_key(ds, *, cfg_seed=0) -> tuple[str, ...]:
    ct = _plan_config_tuple(cfg_seed)
    return (ds, "knn5", "mock.Model", "mock", *ct)


def _linear_row(**overrides) -> dict:
    """A minimal resume-key-bearing linear result row."""
    row = {
        "dataset": "m-eurosat",
        "method": "linear",
        "model": "mock.Model",
        "name": "mock",
        "normalization": "identity",
        "image_size": 8,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
    }
    row.update(overrides)
    return row


def _min_result(**overrides) -> EvaluationResult:
    """Build an EvaluationResult with all required fields, overriding as needed."""
    kwargs = {
        "dataset": "m-eurosat",
        "method": "linear",
        "metric_name": "accuracy",
        "metric_value": 0.9,
        "ci_lower": 0.85,
        "ci_upper": 0.95,
        "feature_dim": 8,
        "best_c": 1.0,
        "best_lr": None,
        "best_batch_size": None,
        "n_train": 100,
        "n_val": 20,
        "n_test": 30,
        "seed": 0,
        "model": "mock.Model",
        "name": "mock",
        "normalization": "identity",
        "image_size": 8,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "c_range_start": -2,
        "c_range_stop": 2,
        "c_range_num": 3,
        "merge_val": False,
        "bootstrap": 10,
    }
    kwargs.update(overrides)
    return EvaluationResult(**kwargs)


class _ImageOnlyDataset(Dataset):
    def __len__(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        del idx
        return {"image": torch.ones(3, 8, 8)}


def test_evaluation_result_row_has_fraction_columns() -> None:
    row = _min_result(train_fraction=0.1, subsample_seed=7, n_train=13).to_row()
    assert row["train_fraction"] == 0.1
    assert row["subsample_seed"] == 7
    assert row["n_train"] == 13


def test_evaluation_result_fraction_defaults() -> None:
    r = _min_result(seed=42)
    assert r.train_fraction == 1.0
    assert r.subsample_seed == 42  # defaults to the row's seed
    row = r.to_row()
    assert row["train_fraction"] == 1.0
    assert row["subsample_seed"] == 42


def test_config_defaults_label_fractions() -> None:
    cfg_path = Path(torchgeo_bench.__file__).parent / "conf" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)
    assert list(cfg.eval.label_fractions) == [1.0]
    assert cfg.eval.n_subsample_repeats == 1


def test_legacy_linear_row_canonicalizes_to_full_fraction() -> None:
    # A legacy CSV with no train_fraction / subsample_seed columns at all.
    legacy = pd.DataFrame([_linear_row()])
    legacy = _normalize_resume_fraction_columns(legacy, base_seed=0)
    completed = _completed_run_keys(legacy, _RESUME_KEY_COLS)
    fresh_key = _row_key(_linear_row(train_fraction=1.0, subsample_seed=0), _RESUME_KEY_COLS)
    assert fresh_key in completed  # legacy row is recognized -> skipped on resume


def test_legacy_empty_cells_canonicalize_to_full_fraction() -> None:
    # Columns present but with empty/NaN cells (partial CSV).
    legacy = pd.DataFrame([_linear_row(train_fraction="", subsample_seed="")])
    legacy = _normalize_resume_fraction_columns(legacy, base_seed=3)
    completed = _completed_run_keys(legacy, _RESUME_KEY_COLS)
    fresh_key = _row_key(_linear_row(train_fraction=1.0, subsample_seed=3), _RESUME_KEY_COLS)
    assert fresh_key in completed


def test_distinct_fractions_distinct_keys() -> None:
    completed = _completed_run_keys(
        pd.DataFrame([_linear_row(train_fraction=1.0, subsample_seed=0)]),
        _RESUME_KEY_COLS,
    )
    key_01 = _row_key(_linear_row(train_fraction=0.1, subsample_seed=0), _RESUME_KEY_COLS)
    assert key_01 not in completed


def test_distinct_subsample_seeds_distinct_keys() -> None:
    r0 = _row_key(_linear_row(train_fraction=0.1, subsample_seed=0), _RESUME_KEY_COLS)
    r1 = _row_key(_linear_row(train_fraction=0.1, subsample_seed=1), _RESUME_KEY_COLS)
    assert r0 != r1
    completed = _completed_run_keys(
        pd.DataFrame([_linear_row(train_fraction=0.1, subsample_seed=0)]),
        _RESUME_KEY_COLS,
    )
    assert r0 in completed
    assert r1 not in completed  # neither draw skips the other


def test_canonical_fraction_string_form() -> None:
    # Whole fractions/seeds collapse identically across config-side and CSV-side forms.
    assert _canonical_key_cell(1.0) == _canonical_key_cell("1.0") == "1"
    assert _canonical_key_cell(0) == _canonical_key_cell("0.0") == "0"
    assert _canonical_key_cell(7) == _canonical_key_cell("7.0") == "7"
    # Non-integer fractions pass through unchanged from both sides.
    assert _canonical_key_cell(0.1) == _canonical_key_cell("0.1") == "0.1"
    assert _canonical_key_cell(0.05) == _canonical_key_cell("0.05") == "0.05"


def test_plan_expands_requested_fraction_seed_pairs() -> None:
    cfg = _plan_config([0.01, 1.0], n_subsample_repeats=2, seed=0)
    plan = _plan_dataset_run(
        cfg=cfg,
        ds_name="m-eurosat",
        ds_cls=_FakeClsDataset,
        knn_k=5,
        seg_method="seg-fpn",
        config_tuple=_plan_config_tuple(0),
        completed_runs=set(),
        completed_metrics={},
    )
    assert set(plan.linear_pairs_to_run) == {(0.01, 0), (0.01, 1), (1.0, 0)}


def test_plan_skip_dataset_only_when_all_pairs_present() -> None:
    cfg = _plan_config([0.01, 1.0], n_subsample_repeats=2, seed=0, resume=True)
    all_linear = {
        _plan_linear_key("m-eurosat", f, s) for (f, s) in [(0.01, 0), (0.01, 1), (1.0, 0)]
    }
    completed = all_linear | {_plan_knn_key("m-eurosat")}

    def _plan(completed_runs):
        return _plan_dataset_run(
            cfg=cfg,
            ds_name="m-eurosat",
            ds_cls=_FakeClsDataset,
            knn_k=5,
            seg_method="seg-fpn",
            config_tuple=_plan_config_tuple(0),
            completed_runs=completed_runs,
            completed_metrics={},
        )

    # All linear pairs + knn present (id/profile disabled) -> whole dataset skipped.
    assert _plan(completed).skip_dataset is True

    # Drop one low-fraction pair -> dataset stays scheduled.
    missing_one = completed - {_plan_linear_key("m-eurosat", 0.01, 1)}
    plan = _plan(missing_one)
    assert plan.skip_dataset is False
    assert (0.01, 1) in plan.linear_pairs_to_run


def test_plan_linear_pairs_filtered_by_resume() -> None:
    cfg = _plan_config([0.01], n_subsample_repeats=2, seed=0, resume=True)
    completed = {_plan_linear_key("m-eurosat", 0.01, 0)}
    plan = _plan_dataset_run(
        cfg=cfg,
        ds_name="m-eurosat",
        ds_cls=_FakeClsDataset,
        knn_k=5,
        seg_method="seg-fpn",
        config_tuple=_plan_config_tuple(0),
        completed_runs=completed,
        completed_metrics={},
    )
    assert list(plan.linear_pairs_to_run) == [(0.01, 1)]


def test_fraction_loop_emits_row_per_pair() -> None:
    x_train, y_train = _cached(60, seed=0)
    x_val, y_val = _cached(30, seed=1)
    x_test, y_test = _cached(20, seed=2)
    pairs = [(0.1, 0), (0.1, 1), (1.0, 0)]
    with mock.patch("torchgeo_bench.main.evaluate_logistic", side_effect=_fake_logistic):
        rows = _call_run_linear_fractions(pairs, x_train, y_train, x_val, y_val, x_test, y_test)
    assert len(rows) == 3
    for (f, s), row in zip(pairs, rows):
        assert row["method"] == "linear"
        assert row["train_fraction"] == f
        assert row["subsample_seed"] == s
        expected_n = len(stratified_subsample_indices(y_train, f, s))
        assert row["n_train"] == expected_n


def test_fraction_loop_forces_merge_val_false() -> None:
    x_train, y_train = _cached(60, seed=0)
    x_val, y_val = _cached(30, seed=1)
    x_test, y_test = _cached(20, seed=2)
    pairs = [(0.1, 0), (1.0, 0)]
    with mock.patch("torchgeo_bench.main.evaluate_logistic", side_effect=_fake_logistic) as m:
        _call_run_linear_fractions(pairs, x_train, y_train, x_val, y_val, x_test, y_test)
    assert m.call_count == 2
    for call in m.call_args_list:
        assert call.kwargs["merge_val"] is False


def test_fraction_loop_slices_val_too() -> None:
    x_train, y_train = _cached(60, seed=0)
    x_val, y_val = _cached(30, seed=1)
    x_test, y_test = _cached(20, seed=2)
    with mock.patch("torchgeo_bench.main.evaluate_logistic", side_effect=_fake_logistic) as m:
        _call_run_linear_fractions([(0.1, 0)], x_train, y_train, x_val, y_val, x_test, y_test)
    # evaluate_logistic(x_train, y_train, x_val, y_val, x_test, y_test, ...)
    passed_x_val = m.call_args_list[0].args[2]
    expected_val_idx = stratified_subsample_indices(y_val, 0.1, 0)
    assert passed_x_val.shape[0] == len(expected_val_idx)
    assert passed_x_val.shape[0] < len(x_val)  # val is actually subsampled


def test_knn_id_profile_only_at_full_fraction() -> None:
    # The linear-fraction loop emits ONLY linear rows; knn/id/profile are not
    # swept per fraction (they run once, outside this helper).
    x_train, y_train = _cached(60, seed=0)
    x_val, y_val = _cached(30, seed=1)
    x_test, y_test = _cached(20, seed=2)
    pairs = [(0.1, 0), (0.1, 1), (1.0, 0)]
    with mock.patch("torchgeo_bench.main.evaluate_logistic", side_effect=_fake_logistic):
        rows = _call_run_linear_fractions(pairs, x_train, y_train, x_val, y_val, x_test, y_test)
    methods = {row["method"] for row in rows}
    assert methods == {"linear"}
    assert not any(row["method"] in {"knn5", "intrinsic_dim", "profile"} for row in rows)


def test_expand_dataset_list_all(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torchgeo_bench.main.list_datasets", lambda: ["m-eurosat", "benv2"])
    assert _expand_dataset_list("all") == ["m-eurosat", "benv2"]


def test_normalize_bands_value_none_and_listconfig() -> None:
    assert _normalize_bands_value(None) == "all"
    cfg_list = OmegaConf.create(["red", "green"])
    assert _normalize_bands_value(cfg_list) == "red,green"


def test_completed_run_keys_metric_name_absent_returns_empty() -> None:
    existing = pd.DataFrame([{"dataset": "m-eurosat", "method": "knn5"}])
    assert _completed_run_keys(existing, ["dataset", "method"], metric_name="accuracy") == set()


def test_filter_completed_metric_rows_partial_filtering() -> None:
    rows = [
        {"dataset": "m-eurosat", "method": "knn5", "metric_name": "accuracy"},
        {"dataset": "m-eurosat", "method": "knn5", "metric_name": "f1"},
    ]
    completed = {"accuracy": {("m-eurosat", "knn5")}}
    filtered = _filter_completed_metric_rows(rows, completed, ["dataset", "method"])
    assert filtered == [{"dataset": "m-eurosat", "method": "knn5", "metric_name": "f1"}]


def test_build_seg_probe_and_solver_rejects_empty_layers() -> None:
    eval_cfg = OmegaConf.create(
        {
            "segmentation": {
                "layers": [],
                "head_type": "fpn",
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss"},
            }
        }
    )
    with pytest.raises(ValueError, match="requires eval.segmentation.layers"):
        _build_seg_probe_and_solver(
            model=torch.nn.Identity(),
            num_classes=2,
            eval_cfg=eval_cfg,
            device=torch.device("cpu"),
            lr=1e-3,
        )


def test_measure_cpu_throughput_budget_exceeded_returns_none_metrics() -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, kernel_size=1), torch.nn.ReLU())
    sample = torch.rand(4, 3, 8, 8)
    metrics = _measure_cpu_throughput(
        model,
        sample,
        cpu_batch_size=2,
        n_warmup=1,
        n_measure=1,
        time_budget_s=0.0,
    )
    assert metrics == {
        "throughput_samples_per_sec": None,
        "latency_ms_per_batch_p50": None,
    }


def test_evaluate_profile_adds_cpu_metrics_branch() -> None:
    loader = DataLoader(_ImageOnlyDataset(), batch_size=2, shuffle=False, num_workers=0)
    common_meta = {
        "dataset": "m-eurosat",
        "seed": 0,
        "model": "mock.Model",
        "name": "mock",
        "normalization": "identity",
        "image_size": 8,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "c_range_start": -2,
        "c_range_stop": 2,
        "c_range_num": 3,
        "merge_val": False,
        "bootstrap": 10,
    }

    with (
        mock.patch(
            "torchgeo_bench.main.measure_profile",
            return_value={"params_m": 0.1, "throughput_samples_per_sec": 20.0},
        ),
        mock.patch(
            "torchgeo_bench.main._measure_cpu_throughput",
            return_value={
                "throughput_samples_per_sec": 3.0,
                "latency_ms_per_batch_p50": 12.0,
            },
        ),
    ):
        rows = evaluate_profile(
            model=torch.nn.Identity(),
            sample_loader=loader,
            device=torch.device("cpu"),
            n_warmup=0,
            n_measure=1,
            common_meta=common_meta,
            feature_dim=8,
            n_counts={"train": 2, "val": 2, "test": 2},
            cpu_throughput_enabled=True,
            cpu_batch_size=2,
            cpu_n_warmup=0,
            cpu_n_measure=1,
            cpu_time_budget_s=1.0,
        )

    metric_names = {row["metric_name"] for row in rows}
    assert "params_m" in metric_names
    assert "throughput_samples_per_sec" in metric_names
    assert "throughput_samples_per_sec_cpu" in metric_names
    assert "latency_ms_per_batch_p50_cpu" in metric_names
