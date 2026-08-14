import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from torchgeo_bench.uq.pipeline import (
    DISTANCE_METRIC_NAMES,
    _RESUME_KEY_COLS,
    _build_resume_set,
    _expected_metrics,
    _is_distance_method,
    _is_uq_classification_dataset,
    _lookup_best_c,
    _normalize_cloud_pattern_mode,
    _run_distance_block,
    _run_uq_block,
    remap_labels_to_probe_classes,
)
from torchgeo_bench.uq.pipeline import (
    main as uq_main,
)


def _resume_key(**overrides: object) -> tuple[str, ...]:
    """Build the expected resume key for :func:`_base_row`'s cell.

    Derived from ``_RESUME_KEY_COLS`` rather than written out positionally: a
    hand-written tuple silently stops matching whenever a key column is added
    (as happened for the two ``svgp_*`` columns, and again for ``init``), which
    is exactly the failure these tests exist to catch.
    """
    values = {**_base_row("accuracy"), "init": "pretrained", **overrides}
    return tuple(str(values.get(col, "")) for col in _RESUME_KEY_COLS)


def _base_row(metric_name: str) -> dict[str, object]:
    return {
        "model": "m.t",
        "name": "resnet50",
        "seed": 42,
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "uq_method": "uncalibrated",
        "corruption_type": "clean",
        "severity": 0,
        "metric_name": metric_name,
        "metric_value": 0.1,
    }


def test_build_resume_set_empty_csv(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    assert _build_resume_set(str(csv_path)) == set()


def test_build_resume_set_complete_key(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    rows = [_base_row(metric) for metric in sorted(_expected_metrics("uncalibrated"))]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    done = _build_resume_set(str(csv_path))
    assert _resume_key() in done


def test_build_resume_set_partial_key(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    metrics = sorted(_expected_metrics("uncalibrated"))
    rows = [_base_row(metric) for metric in metrics[:-1]]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    done = _build_resume_set(str(csv_path))
    assert _resume_key() not in done


def test_expected_metrics_for_method():
    assert _expected_metrics("uncalibrated") == {
        "accuracy",
        "ece",
        "signed_ece",
        "nll",
        "brier",
        "predictive_entropy",
        "normalized_predictive_entropy",
        "max_probability",
        "raw_aurc",
        "eaurc",
        "selective_acc_90",
    }
    assert _expected_metrics("conformal") == {
        "accuracy",
        "empirical_coverage",
        "mean_set_size",
    }
    assert _expected_metrics("svgp") == {
        "accuracy",
        "ece",
        "signed_ece",
        "nll",
        "brier",
        "predictive_entropy",
        "normalized_predictive_entropy",
        "max_probability",
        "raw_aurc",
        "eaurc",
        "selective_acc_90",
    }


def test_is_uq_classification_dataset():
    class _SingleLabelCls:
        task = "classification"
        multilabel = False

    class _MultiLabelCls:
        task = "classification"
        multilabel = True

    class _Segmentation:
        task = "segmentation"
        multilabel = False

    assert _is_uq_classification_dataset(_SingleLabelCls)
    assert not _is_uq_classification_dataset(_MultiLabelCls)
    assert not _is_uq_classification_dataset(_Segmentation)


def test_lookup_best_c_returns_direct_match():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.25,
            }
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms",
            "partition": "default",
            "bands": "rgb",
        },
    )
    assert got == 0.25


def test_lookup_best_c_no_match_no_alias():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.25,
            }
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms_cr_c1",
            "partition": "default",
            "bands": "rgb",
        },
    )
    assert got is None


def test_lookup_best_c_falls_back_to_alias():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.75,
            }
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms_cr_c3",
            "partition": "default",
            "bands": "rgb",
        },
        alias_dataset="sen12ms",
    )
    assert got == 0.75


def test_lookup_best_c_direct_takes_precedence():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.75,
            },
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms_cr_c3",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 1.5,
            },
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms_cr_c3",
            "partition": "default",
            "bands": "rgb",
        },
        alias_dataset="sen12ms",
    )
    assert got == 1.5


def test_lookup_best_c_alias_logs_info(caplog):
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.5,
            }
        ]
    )
    with caplog.at_level("INFO"):
        got = _lookup_best_c(
            prior,
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms_cr_c4",
                "partition": "default",
                "bands": "rgb",
            },
            alias_dataset="sen12ms",
        )
    assert got == 0.5
    assert "using alias" in caplog.text.lower()


def test_lookup_best_c_falls_back_to_alias_sweep_c_format():
    prior = pd.DataFrame(
        [
            {"dataset": "sen12ms", "model": "resnet50", "C": 0.9, "val_acc": 0.80},
            {"dataset": "sen12ms", "model": "resnet50", "C": 0.7, "val_acc": 0.85},
        ]
    )
    got = _lookup_best_c(
        prior,
        {"dataset": "sen12ms_cr_c2", "name": "resnet50"},
        alias_dataset="sen12ms",
    )
    assert got == 0.7


def test_pipeline_passes_alias_to_lookup(monkeypatch, tmp_path):
    csv_path = tmp_path / "all_results.csv"
    pd.DataFrame([{"method": "linear", "best_c": 0.3}]).to_csv(csv_path, index=False)

    class _DummyBench:
        task = "classification"
        multilabel = False
        prior_results_alias = "sen12ms"
        rgb_bands = ["red", "green", "blue"]

        def select_band_specs(self, bands):  # noqa: ANN001
            return []

    class _DummyModel:
        def to(self, _device):  # noqa: ANN001
            return self

        def eval(self):
            return self

    seen_alias: list[str | None] = []

    def _fake_lookup(prior_results, row_filter, *, alias_dataset=None):  # noqa: ARG001
        seen_alias.append(alias_dataset)
        return None

    monkeypatch.setattr("torchgeo_bench.uq.pipeline.get_bench_dataset_class", lambda _name: _DummyBench)
    monkeypatch.setattr(
        "torchgeo_bench.uq.pipeline.get_datasets",
        lambda **kwargs: (object(), object(), object(), object()),
    )
    monkeypatch.setattr("torchgeo_bench.uq.pipeline.instantiate", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(
        "torchgeo_bench.uq.pipeline.extract_features",
        lambda *args, **kwargs: (
            np.zeros((8, 4), dtype=np.float32),
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr("torchgeo_bench.uq.pipeline._lookup_best_c", _fake_lookup)

    cfg = OmegaConf.create(
        {
            "seed": 1,
            "resume": False,
            "device": "cpu",
            "verbose": False,
            "model": {"_target_": "dummy.Target", "name": "resnet50"},
            "dataset": {
                "names": ["sen12ms_cr_c1"],
                "partition": "default",
                "batch_size": 2,
                "num_workers": 0,
                "bands": "rgb",
                "interpolation": "bilinear",
                "normalization": "bandspec_zscore",
            },
            "uq": {
                "output": str(tmp_path / "uq_results.csv"),
                "prior_results": str(csv_path),
                "methods": ["uncalibrated"],
                "corruptions": ["clean"],
                "corruption_severities": [1],
                "cal_size": 2,
                "ece_bins": 10,
                "ece_binning": "equal_width",
                "conformal_alpha": 0.1,
                "n_ensemble": 2,
                "laplace_batch_size": 16,
                "cloud_pattern_mode": "fixed_across_severity",
            },
        }
    )
    uq_main.__wrapped__(cfg)  # type: ignore[attr-defined]
    assert seen_alias == ["sen12ms"]


def test_run_uq_block_writes_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "uq_results.csv"
    X_test = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0, 1, 2, 1], dtype=np.int64)

    def _fake_extract(*args, **kwargs):  # noqa: ARG001
        return X_test, y_test

    monkeypatch.setattr("torchgeo_bench.uq.pipeline.extract_features", _fake_extract)

    class _DummyUncalibrated:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            logits = X.copy()
            logits = logits - logits.max(axis=1, keepdims=True)
            exps = np.exp(logits)
            return exps / exps.sum(axis=1, keepdims=True)

    common_meta = {
        "model": "m.t",
        "name": "resnet50",
        # Mirrors the pipeline's common_meta, which carries the backbone-init
        # arm so the two arms get distinct resume keys.
        "init": "pretrained",
        "backbone": "resnet50",
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "seed": 42,
    }

    rows = _run_uq_block(
        method_name="uncalibrated",
        method=_DummyUncalibrated(),
        output_path=str(csv_path),
        common_meta=common_meta,
        corruption_type="clean",
        severity=0,
        ece_bins=15,
        ece_binning="equal_width",
        conformal_alpha=0.1,
        n_cal=40,
        n_train=160,
        feature_dim=3,
        best_c=1.0,
        seed=42,
        model=object(),  # type: ignore[arg-type]
        test_loader=object(),  # type: ignore[arg-type]
        verbose=False,
    )

    df = pd.read_csv(csv_path)
    assert len(rows) == 10
    assert set(df["metric_name"]) == _expected_metrics("uncalibrated")
    assert np.isfinite(df["metric_value"].to_numpy(dtype=np.float64)).all()


def test_run_uq_block_conformal_writes_reduced_metrics(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    X_test = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0, 1, 2, 1], dtype=np.int64)

    class _DummyConformal:
        def predict_sets(self, X: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
            assert alpha == 0.1
            point_preds = np.array([0, 0, 2, 1], dtype=np.int64)
            pred_sets = np.array(
                [
                    [True, False, False],
                    [True, True, False],
                    [False, False, True],
                    [False, True, False],
                ],
                dtype=bool,
            )
            assert X.shape == X_test.shape
            return point_preds, pred_sets

    common_meta = {
        "model": "m.t",
        "name": "resnet50",
        # Mirrors the pipeline's common_meta, which carries the backbone-init
        # arm so the two arms get distinct resume keys.
        "init": "pretrained",
        "backbone": "resnet50",
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "seed": 42,
    }

    rows = _run_uq_block(
        method_name="conformal",
        method=_DummyConformal(),
        output_path=str(csv_path),
        common_meta=common_meta,
        corruption_type="clean",
        severity=0,
        ece_bins=15,
        ece_binning="equal_width",
        conformal_alpha=0.1,
        n_cal=40,
        n_train=160,
        feature_dim=3,
        best_c=1.0,
        seed=42,
        X_test=X_test,
        y_test=y_test,
    )

    df = pd.read_csv(csv_path)
    assert len(rows) == 3
    assert set(df["metric_name"]) == _expected_metrics("conformal")


def test_normalize_cloud_pattern_mode():
    assert _normalize_cloud_pattern_mode("fixed_across_severity") == "fixed"
    assert _normalize_cloud_pattern_mode("independent_per_severity") == "independent"
    assert _normalize_cloud_pattern_mode("fixed") == "fixed"
    assert _normalize_cloud_pattern_mode("independent") == "independent"


def test_normalize_cloud_pattern_mode_invalid():
    with np.testing.assert_raises(ValueError):
        _normalize_cloud_pattern_mode("invalid")


def test_expected_metrics_nf_empirical():
    assert _expected_metrics("nf_empirical") == _expected_metrics("uncalibrated")


def test_expected_metrics_nf_uniform():
    assert _expected_metrics("nf_uniform") == _expected_metrics("uncalibrated")


def test_run_uq_block_nf_empirical_writes_standard_metrics(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    X_test = np.random.default_rng(0).standard_normal((12, 4)).astype(np.float32)
    y_test = np.tile([0, 1, 2], 4).astype(np.int64)

    class _DummyNF:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.full((len(X), 3), 1 / 3, dtype=np.float32)

        def predict_confidence(self, X: np.ndarray) -> np.ndarray:
            return np.zeros(len(X), dtype=np.float32)

    rows = _run_uq_block(
        method_name="nf_empirical",
        method=_DummyNF(),
        output_path=str(csv_path),
        common_meta={
            "model": "m.t", "name": "resnet50", "backbone": "resnet50",
            "dataset": "m-eurosat", "normalization": "bandspec_zscore",
            "image_size": 224, "interpolation": "bilinear",
            "partition": "default", "bands": "rgb", "seed": 42,
        },
        corruption_type="clean", severity=0, ece_bins=15,
        ece_binning="equal_width", conformal_alpha=0.1,
        n_cal=0, n_train=100, feature_dim=4,
        best_c=float("nan"), seed=42,
        X_test=X_test, y_test=y_test,
    )
    df = pd.read_csv(csv_path)
    assert set(df["metric_name"]) == _expected_metrics("nf_empirical")
    assert np.isfinite(df["metric_value"].to_numpy(dtype=np.float64)).all()


# --------------------------------------------------------------------------
# Activation-distance deferral (docs/plans/activation_distance_deferral.md)
# --------------------------------------------------------------------------


class _StubProbe:
    """Minimal probe stand-in: fixed probabilities plus a ``classes_`` vector."""

    def __init__(self, probs: np.ndarray, classes: np.ndarray | None = None) -> None:
        self._probs = probs
        if classes is not None:
            self.classes_ = classes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probs[: len(X)]


class _StubReference:
    """Activation reference stand-in returning a caller-supplied score vector."""

    def __init__(self, scores: np.ndarray, layer_name: str = "backbone.layer4", kind: str = "maha") -> None:
        self._scores = np.asarray(scores, dtype=np.float64)
        self.layer_name = layer_name
        self.score_kind = kind

    @property
    def key(self) -> str:
        return f"{self.score_kind}@{self.layer_name}"

    def score(self, acts: np.ndarray) -> np.ndarray:
        return self._scores


def _distance_common_meta() -> dict[str, object]:
    return {
        "model": "m.t",
        "name": "resnet50",
        # Mirrors the pipeline's common_meta, which carries the backbone-init
        # arm so the two arms get distinct resume keys.
        "init": "pretrained",
        "backbone": "resnet50",
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "seed": 42,
    }


def _run_distance_fixture(tmp_path, *, scores, probs, y_test, classes=None, trace_ctx=None, kind="maha"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_path / "uq_results.csv"
    n = len(y_test)
    rows = _run_distance_block(
        reference=_StubReference(scores, kind=kind),
        acts_test=np.zeros((n, 3), dtype=np.float32),
        probe=_StubProbe(probs, classes),
        output_path=str(csv_path),
        common_meta=_distance_common_meta(),
        corruption_type="poisson_gaussian",
        severity=3,
        n_cal=0,
        n_train=100,
        feature_dim=4,
        best_c=1.0,
        X_test=np.zeros((n, 4), dtype=np.float32),
        y_test=np.asarray(y_test, dtype=np.int64),
        sample_ids=np.array([f"s{i}" for i in range(n)], dtype=object),
        trace_ctx=trace_ctx,
    )
    return csv_path, rows


def test_expected_metrics_for_distance_methods():
    for name in ("maha@backbone.layer4", "maha_classagnostic@backbone.blocks.11", "knn@rcf"):
        assert _is_distance_method(name)
        assert _expected_metrics(name) == DISTANCE_METRIC_NAMES
    assert not _is_distance_method("uncalibrated")
    assert not _is_distance_method("conformal")


def test_run_distance_block_emits_exactly_the_ranking_metrics(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    probs = rng.dirichlet(np.ones(3), size=n)
    y_test = rng.integers(0, 3, size=n)
    scores = rng.random(n)

    csv_path, rows = _run_distance_fixture(tmp_path, scores=scores, probs=probs, y_test=y_test)

    df = pd.read_csv(csv_path)
    assert set(df["metric_name"]) == DISTANCE_METRIC_NAMES
    # No probabilistic metrics leak in.
    assert not (set(df["metric_name"]) & {"ece", "nll", "brier", "accuracy"})
    assert len(rows) == len(DISTANCE_METRIC_NAMES)
    assert set(df["uq_method"]) == {"maha@backbone.layer4"}
    assert np.isfinite(df["metric_value"].to_numpy(dtype=np.float64)).all()


def test_run_distance_block_resume_roundtrip_marks_cell_complete(tmp_path):
    """Emit -> rebuild the resume set -> the cell must be complete.

    Guards the error_auroc contract end to end: checking _expected_metrics'
    return value in isolation would pass even while resume is broken, because
    the failure mode is the emitter and the expectation disagreeing.
    """
    rng = np.random.default_rng(1)
    n = 30
    probs = rng.dirichlet(np.ones(3), size=n)
    y_test = rng.integers(0, 3, size=n)
    csv_path, _ = _run_distance_fixture(tmp_path, scores=rng.random(n), probs=probs, y_test=y_test)

    done = _build_resume_set(str(csv_path))
    # Build the key from _RESUME_KEY_COLS rather than hardcoding a tuple, so the
    # test tracks the key schema instead of a snapshot of its current width.
    values = {
        **_distance_common_meta(),
        "uq_method": "maha@backbone.layer4",
        "corruption_type": "poisson_gaussian",
        "severity": 3,
    }
    key = tuple(str(values.get(col, "")) for col in _RESUME_KEY_COLS)
    assert key in done, (sorted(done), key)


def test_run_distance_block_remap_matches_run_uq_block(tmp_path):
    """The two blocks must agree on label indexing when classes_ is not [0..C-1].

    y_pred/correct come from the probe, so a disagreement here would make the
    distance rows silently describe different samples than the uncalibrated rows
    they are joined against in analysis -- and the analysis row-count assertion
    would still pass.
    """
    classes = np.array([2, 5, 7], dtype=np.int64)
    y_raw = np.array([2, 5, 7, 2, 5, 7, 7, 2], dtype=np.int64)
    probs = np.tile(np.array([[0.7, 0.2, 0.1]]), (len(y_raw), 1))

    assert remap_labels_to_probe_classes(y_raw, _StubProbe(probs, classes)).tolist() == [
        0, 1, 2, 0, 1, 2, 2, 0
    ]

    dist_csv, _ = _run_distance_fixture(
        tmp_path / "dist",
        scores=np.arange(len(y_raw), dtype=np.float64),
        probs=probs,
        y_test=y_raw,
        classes=classes,
    )
    (tmp_path / "uq").mkdir(parents=True, exist_ok=True)
    uq_csv = tmp_path / "uq" / "uq_results.csv"
    _run_uq_block(
        method_name="uncalibrated",
        method=_StubProbe(probs, classes),
        output_path=str(uq_csv),
        common_meta=_distance_common_meta(),
        corruption_type="poisson_gaussian",
        severity=3,
        ece_bins=15,
        ece_binning="equal_width",
        conformal_alpha=0.1,
        n_cal=0,
        n_train=100,
        feature_dim=4,
        best_c=1.0,
        seed=42,
        X_test=np.zeros((len(y_raw), 4), dtype=np.float32),
        y_test=y_raw.copy(),
    )

    dist_df = pd.read_csv(dist_csv)
    uq_df = pd.read_csv(uq_csv)
    # Both blocks see the same 3-of-8 correct predictions (class index 0).
    acc = float(uq_df.loc[uq_df["metric_name"] == "accuracy", "metric_value"].iloc[0])
    assert acc == 3 / 8
    assert set(dist_df["metric_name"]) == DISTANCE_METRIC_NAMES


def test_run_distance_block_sign_convention(tmp_path):
    """A score correlated with error must beat one anti-correlated with it.

    Distance is an uncertainty and _risk_coverage_curve sorts by DESCENDING
    confidence, so it enters as confidence = -distance. Getting that backwards
    inverts the curve and yields a spectacular-looking spurious result.
    """
    n = 40
    # First half correct (argmax 0 == y), second half wrong.
    probs = np.tile(np.array([[0.6, 0.4]]), (n, 1))
    y_test = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.int64)
    aligned = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])  # high distance <-> error
    anti = 1.0 - aligned

    good_csv, _ = _run_distance_fixture(tmp_path / "good", scores=aligned, probs=probs, y_test=y_test)
    bad_csv, _ = _run_distance_fixture(tmp_path / "bad", scores=anti, probs=probs, y_test=y_test)

    def _metric(path, name):
        df = pd.read_csv(path)
        return float(df.loc[df["metric_name"] == name, "metric_value"].iloc[0])

    assert _metric(good_csv, "eaurc") < _metric(bad_csv, "eaurc")
    assert _metric(good_csv, "error_auroc") == 1.0
    assert _metric(bad_csv, "error_auroc") == 0.0
    assert _metric(good_csv, "selective_acc_90") > _metric(bad_csv, "selective_acc_90")


def test_run_distance_block_degenerate_error_auroc_still_emits_row(tmp_path):
    """An all-correct block has undefined AUROC but must still resume as complete."""
    n = 10
    probs = np.tile(np.array([[0.9, 0.1]]), (n, 1))
    y_test = np.zeros(n, dtype=np.int64)  # every prediction correct
    csv_path, _ = _run_distance_fixture(
        tmp_path, scores=np.arange(n, dtype=np.float64), probs=probs, y_test=y_test
    )
    df = pd.read_csv(csv_path)
    assert set(df["metric_name"]) == DISTANCE_METRIC_NAMES
    assert np.isnan(df.loc[df["metric_name"] == "error_auroc", "metric_value"].iloc[0])
    assert _build_resume_set(str(csv_path))


def test_run_distance_block_rejects_misaligned_activations(tmp_path):
    n = 12
    probs = np.tile(np.array([[0.6, 0.4]]), (n, 1))
    with np.testing.assert_raises(ValueError):
        _run_distance_block(
            reference=_StubReference(np.zeros(n)),
            acts_test=np.zeros((n - 3, 5), dtype=np.float32),  # row-misaligned
            probe=_StubProbe(probs),
            output_path=str(tmp_path / "uq.csv"),
            common_meta=_distance_common_meta(),
            corruption_type="clean",
            severity=0,
            n_cal=0,
            n_train=10,
            feature_dim=4,
            best_c=1.0,
            X_test=np.zeros((n, 4), dtype=np.float32),
            y_test=np.zeros(n, dtype=np.int64),
        )


def test_distance_blocks_land_in_distinct_trace_partitions(tmp_path):
    from torchgeo_bench.uq.traces import scan_traces

    rng = np.random.default_rng(2)
    n = 16
    probs = rng.dirichlet(np.ones(3), size=n)
    y_test = rng.integers(0, 3, size=n)
    root = tmp_path / "uq_traces"
    trace_ctx = {
        "run_id": "run-1",
        "trace_dataset_root": str(root),
        "config_hash": "cfg",
        "git_sha": "sha",
        "created_at_utc": "2026-08-11T00:00:00Z",
        "compression": "zstd",
        "overwrite": False,
        "include_conformal": False,
    }
    for kind in ("maha", "maha_classagnostic", "knn"):
        _run_distance_fixture(
            tmp_path,
            scores=rng.random(n),
            probs=probs,
            y_test=y_test,
            trace_ctx=trace_ctx,
            kind=kind,
        )

    partitions = sorted(p.name for p in (root / "dataset=m-eurosat" / "backbone=resnet50").iterdir())
    assert partitions == [
        "uq_method=knn@backbone.layer4",
        "uq_method=maha@backbone.layer4",
        "uq_method=maha_classagnostic@backbone.layer4",
    ]
    scanned = scan_traces(str(root), columns=["uq_method", "score_kind", "distance_score"])
    assert len(scanned) == 3 * n
    assert set(scanned["score_kind"]) == {"maha", "maha_classagnostic", "knn"}


def test_single_layer_model_produces_one_layer_of_partitions(tmp_path):
    """rcf_empirical has a single hook path; nothing may assume four layers."""
    from torchgeo_bench.uq.distance import build_activation_references
    from torchgeo_bench.uq.traces import scan_traces

    rng = np.random.default_rng(3)
    n_train, n_test, d = 60, 16, 5
    acts_train = {"rcf": rng.normal(size=(n_train, d))}
    y_train = rng.integers(0, 3, size=n_train)
    refs = build_activation_references(acts_train=acts_train, y_train=y_train, knn_k=5)

    probs = rng.dirichlet(np.ones(3), size=n_test)
    y_test = rng.integers(0, 3, size=n_test)
    root = tmp_path / "uq_traces"
    trace_ctx = {
        "run_id": "run-1",
        "trace_dataset_root": str(root),
        "config_hash": "cfg",
        "git_sha": "sha",
        "created_at_utc": "2026-08-11T00:00:00Z",
        "compression": "zstd",
        "overwrite": False,
        "include_conformal": False,
    }
    for ref in refs.values():
        _run_distance_block(
            reference=ref,
            acts_test=rng.normal(size=(n_test, d)),
            probe=_StubProbe(probs),
            output_path=str(tmp_path / "uq.csv"),
            common_meta=_distance_common_meta(),
            corruption_type="clean",
            severity=0,
            n_cal=0,
            n_train=n_train,
            feature_dim=d,
            best_c=1.0,
            X_test=np.zeros((n_test, d), dtype=np.float32),
            y_test=y_test,
            trace_ctx=trace_ctx,
        )

    scanned = scan_traces(str(root), columns=["uq_method", "layer_name"])
    assert set(scanned["layer_name"]) == {"rcf"}
    assert set(scanned["uq_method"]) == {"maha@rcf", "maha_classagnostic@rcf", "knn@rcf"}


def test_build_resume_key_matches_build_resume_set_schema(tmp_path):
    """A key built for lookup must match the width and content of a stored key.

    Regression: _resume_key spelled out a 12-tuple positionally while
    _RESUME_KEY_COLS had grown to 14 (the two svgp_* columns), so the lookup
    never matched a stored key and every cell silently re-ran on resume.
    """
    from omegaconf import OmegaConf

    from torchgeo_bench.uq.pipeline import build_resume_key

    common_meta = _distance_common_meta() | {
        "svgp_inducing_init": "kmeans",
        "svgp_mixing_weights": False,
    }
    rng = np.random.default_rng(7)
    n = 20
    csv_path, _ = _run_distance_fixture(
        tmp_path,
        scores=rng.random(n),
        probs=rng.dirichlet(np.ones(3), size=n),
        y_test=rng.integers(0, 3, size=n),
    )
    # _run_distance_block writes only common_meta, so add the svgp columns the
    # real pipeline's common_meta carries.
    df = pd.read_csv(csv_path)
    for col, val in (("svgp_inducing_init", "kmeans"), ("svgp_mixing_weights", False)):
        df[col] = val
    df.to_csv(csv_path, index=False)

    cfg = OmegaConf.create(
        {
            "model": {"_target_": "m.t", "name": "resnet50"},
            "seed": 42,
            "dataset": {"image_size": 224, "interpolation": "bilinear", "partition": "default"},
        }
    )
    key = build_resume_key(
        common_meta=common_meta,
        cfg=cfg,
        dataset_name="m-eurosat",
        normalization="bandspec_zscore",
        bands_value="rgb",
        uq_method="maha@backbone.layer4",
        corruption_type="poisson_gaussian",
        severity="3",
    )
    completed = _build_resume_set(str(csv_path))
    assert len(key) == len(_RESUME_KEY_COLS)
    assert key in completed, (sorted(completed), key)
