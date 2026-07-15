"""Smoke + unit tests for ``experiments/scripts/regen_leaderboard.py``.

The generator is a standalone script (not an installed module), so we import it
by inserting ``experiments/scripts`` on ``sys.path`` — mirroring the convention
in ``test_geofm_cka_prototypes_smoke.py``. Tests that need evaluma skip
gracefully when it is not installed.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import regen_leaderboard as rl  # noqa: E402

# --- synthetic raw frame ---------------------------------------------------

SINGLE_LABEL = ["m-eurosat", "m-so2sat", "m-forestnet"]  # metric accuracy
MULTI_LABEL = ["m-bigearthnet", "treesatai"]  # metric micro_mAP
DATASETS = SINGLE_LABEL + MULTI_LABEL
NORMALIZATIONS = ("bandspec_zscore", "model_native")

# Base quality per model; higher = better. Baselines sit at the bottom.
BASE_SCORE = {"good": 0.90, "mid": 0.80, "low": 0.70, "rcf": 0.45, "imagestats": 0.40}
CLS_MODELS = {"good", "mid", "low"}
MS_BANDS_A = "blue,green,red,nir,swir_1,swir_2"

# Make the normalization axis materially reorder the board.
NORM_OFFSET = {
    "bandspec_zscore": {"good": 0.03, "mid": 0.00, "low": -0.01, "rcf": 0.00, "imagestats": 0.00},
    "model_native": {"good": -0.04, "mid": 0.05, "low": 0.01, "rcf": 0.00, "imagestats": 0.00},
}


def _metric_for(dataset: str) -> str:
    return "accuracy" if dataset in SINGLE_LABEL else "micro_mAP"


def _score(
    base: str,
    dataset: str,
    probe: str,
    bandclass: str,
    pooling: str,
    normalization: str,
) -> float:
    s = BASE_SCORE[base]
    s += 0.0 if probe == "linear" else -0.03
    s += 0.02 if bandclass == "Multispectral" else 0.0
    s += -0.01 if pooling == "cls-token" else 0.0
    s += NORM_OFFSET[normalization][base]
    s += 0.001 * DATASETS.index(dataset)  # deterministic per-dataset jitter
    return round(s, 4)


def _row(
    name: str,
    dataset: str,
    probe: str,
    bands: str,
    score: float,
    normalization: str,
) -> dict[str, object]:
    return {
        "name": name,
        "dataset": dataset,
        "method": probe,
        "metric_name": _metric_for(dataset),
        "metric_value": score,
        "bands": bands,
        "normalization": normalization,
    }


def _raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base in BASE_SCORE:
        for dataset in DATASETS:
            for probe in ("linear", "knn5"):
                for normalization in NORMALIZATIONS:
                    rows.append(
                        _row(
                            base,
                            dataset,
                            probe,
                            "rgb",
                            _score(base, dataset, probe, "RGB", "patch-mean", normalization),
                            normalization,
                        )
                    )
                    rows.append(
                        _row(
                            base,
                            dataset,
                            probe,
                            MS_BANDS_A,
                            _score(
                                base,
                                dataset,
                                probe,
                                "Multispectral",
                                "patch-mean",
                                normalization,
                            ),
                            normalization,
                        )
                    )
                    if base in CLS_MODELS:
                        rows.append(
                            _row(
                                f"{base}_cls",
                                dataset,
                                probe,
                                "rgb",
                                _score(base, dataset, probe, "RGB", "cls-token", normalization),
                                normalization,
                            )
                        )
                        rows.append(
                            _row(
                                f"{base}_cls",
                                dataset,
                                probe,
                                MS_BANDS_A,
                                _score(
                                    base,
                                    dataset,
                                    probe,
                                    "Multispectral",
                                    "cls-token",
                                    normalization,
                                ),
                                normalization,
                            )
                        )

    # A probe-asymmetric model: complete under RGB/linear/patch-mean but with no
    # knn5 rows, so the probe-axis intersection is strictly smaller than the
    # linear roster.
    for dataset in DATASETS:
        for normalization in NORMALIZATIONS:
            rows.append(
                _row(
                    "lin_only",
                    dataset,
                    "linear",
                    "rgb",
                    _score("mid", dataset, "linear", "RGB", "patch-mean", normalization),
                    normalization,
                )
            )

    # OlmoEarth-style single-dataset ablation rows — must be dropped.
    for suffix in ("model_lsat_x100", "model_naip_rgb", "model_sar_db"):
        rows.append(_row(suffix, "m-eurosat", "linear", "rgb", 0.5, "bandspec_zscore"))

    # A partial-coverage model: present in every RGB linear patch-mean dataset
    # except one — should be excluded from that slice, not seated.
    for dataset in DATASETS:
        if dataset == "treesatai":
            continue
        for normalization in NORMALIZATIONS:
            rows.append(
                _row(
                    "partial",
                    dataset,
                    "linear",
                    "rgb",
                    _score("mid", dataset, "linear", "RGB", "patch-mean", normalization),
                    normalization,
                )
            )

    return rows


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(_raw_rows())


def _axes() -> dict[str, tuple[str, str, str]]:
    return rl.sensitivity_axes(list(NORMALIZATIONS))


def _slice_block(rankings: dict, slice_key: dict) -> dict:
    return rankings[slice_key["task"]][slice_key["probe"]][slice_key["bandclass"]][
        slice_key["pooling"]
    ][slice_key["normalization"]]


# --- Slice 1 tests ---------------------------------------------------------


def test_drops_ablation_rows():
    out = rl.harmonize(_raw_df())
    for tag in ("_lsat_", "_naip_", "_sar_"):
        assert not out["base_model"].astype(str).str.contains(tag).any()


def test_harmonize_drops_excluded_datasets():
    raw = _raw_df()
    dropped = next(iter(rl.DROPPED_DATASETS))
    extra = _row("good", dropped, "linear", "rgb", 0.99, "bandspec_zscore")
    raw = pd.concat([raw, pd.DataFrame([extra])], ignore_index=True)
    out = rl.harmonize(raw)
    assert rl.DROPPED_DATASETS.isdisjoint(set(out["dataset"].unique()))


def test_derives_pooling_and_base_model():
    out = rl.harmonize(_raw_df())
    cls = out[out["pooling"] == "cls-token"]
    assert not cls.empty
    assert (cls["base_model"] == "good").any()
    assert not cls["base_model"].astype(str).str.endswith("_cls").any()
    patch = out[out["base_model"] == "good"]
    assert (patch["pooling"] == "patch-mean").any()


def test_derives_bandclass():
    out = rl.harmonize(_raw_df())
    assert set(out["bandclass"].unique()) == {"RGB", "Multispectral"}
    rgb = out[out["bandclass"] == "RGB"]
    assert not rgb.empty


def test_harmonize_keeps_normalization_axis():
    out = rl.harmonize(_raw_df())
    assert set(out["normalization"].unique()) == set(NORMALIZATIONS)


def test_canonical_metric_map():
    out = rl.harmonize(_raw_df())
    assert set(out["metric"].unique()) == {"accuracy", "map"}


def test_does_not_collapse_across_normalization():
    out = rl.harmonize(_raw_df())
    cell = out[
        (out["base_model"] == "good")
        & (out["dataset"] == "m-eurosat")
        & (out["probe"] == "linear")
        & (out["bandclass"] == "RGB")
        & (out["pooling"] == "patch-mean")
    ]
    assert len(cell) == 2
    by_norm = {row.normalization: row.score for row in cell.itertuples(index=False)}
    assert by_norm["bandspec_zscore"] == pytest.approx(
        _score("good", "m-eurosat", "linear", "RGB", "patch-mean", "bandspec_zscore")
    )
    assert by_norm["model_native"] == pytest.approx(
        _score("good", "m-eurosat", "linear", "RGB", "patch-mean", "model_native")
    )


def test_model_meta_resolver():
    assert rl.resolve_meta("rcf") == ("Random conv. features", "Baseline", True)
    assert rl.resolve_meta("imagestats")[2] is True
    assert rl.resolve_meta("totally_unknown_slug") == (
        "totally_unknown_slug",
        "totally_unknown_slug",
        False,
    )


# --- D8: measured params column --------------------------------------------


def _params_row(name: str, value: float) -> dict[str, object]:
    return {
        "name": name,
        "dataset": "m-eurosat",
        "method": "linear",
        "metric_name": "params_m",
        "metric_value": value,
        "bands": "rgb",
        "normalization": "bandspec_zscore",
    }


def test_extract_params_collapses_strips_cls_and_drops_ablations():
    raw = pd.DataFrame(
        [
            _params_row("good", 300.4),
            _params_row("good", 304.3),  # cross-band spread -> max wins
            _params_row("good_cls", 304.3),  # _cls suffix -> same base model
            _params_row("rcf", 0.0),  # baseline recorded as 0
            _params_row("model_lsat_x100", 42.0),  # ablation -> dropped
        ]
    )
    params = rl.extract_params(raw)
    assert params["good"] == pytest.approx(304.3)
    assert params["rcf"] == pytest.approx(0.0)
    assert not any("_lsat_" in model for model in params)
    assert "good_cls" not in params


def test_model_meta_carries_params_m():
    pytest.importorskip("evaluma")
    raw = _raw_df()
    raw = pd.concat(
        [raw, pd.DataFrame([_params_row("good", 304.3), _params_row("mid", 87.0)])],
        ignore_index=True,
    )
    harmonized = rl.harmonize(raw)
    params = rl.extract_params(raw)
    _rankings, _sensitivity, meta, _default = rl.assemble(
        harmonized, n_bootstrap=50, params=params, random_state=0
    )
    assert meta["good"]["params_m"] == pytest.approx(304.3)
    assert meta["low"]["params_m"] is None


# --- Slice 2 tests ---------------------------------------------------------

DEFAULT_SLICE_KEY = {
    "task": "classification",
    "probe": "linear",
    "bandclass": "RGB",
    "pooling": "patch-mean",
    "normalization": "bandspec_zscore",
}


def test_build_slice_complete_matrix():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    scores = bench.scores_
    assert not scores.isna().to_numpy().any()
    assert set(bench.datasets_) == set(DATASETS)


def test_build_slice_records_excluded():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    assert "partial" not in set(bench.models_)
    entry = next(e for e in excluded if e["model"] == "partial")
    assert entry["n_tasks"] == len(DATASETS) - 1


def test_build_slice_keeps_baselines():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    assert {"rcf", "imagestats"}.issubset(set(bench.models_))


def test_build_slice_uses_requested_normalization():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench_z, _ = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    bench_native, _ = rl.build_slice(
        harmonized, **{**DEFAULT_SLICE_KEY, "normalization": "model_native"}
    )
    assert bench_z is not None and bench_native is not None
    assert bench_z._raw.loc["good", "m-eurosat"] == pytest.approx(
        _score("good", "m-eurosat", "linear", "RGB", "patch-mean", "bandspec_zscore")
    )
    assert bench_native._raw.loc["good", "m-eurosat"] == pytest.approx(
        _score("good", "m-eurosat", "linear", "RGB", "patch-mean", "model_native")
    )
    assert bench_z._raw.loc["good", "m-eurosat"] != bench_native._raw.loc["good", "m-eurosat"]


def test_build_slice_rejects_duplicate_seated_cells():
    pytest.importorskip("evaluma")
    raw = pd.concat(
        [
            _raw_df(),
            pd.DataFrame(
                [
                    _row(
                        "good",
                        "m-eurosat",
                        "linear",
                        "rgb",
                        0.1234,
                        "bandspec_zscore",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    harmonized = rl.harmonize(raw)
    with pytest.raises(ValueError, match="Duplicate seated leaderboard cells"):
        rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)


def _bench_from_scores(score_map: dict[str, dict[str, float]], metric: str = "accuracy"):
    """Build a dense evaluma Benchmark from ``{model: {dataset: score}}``."""
    import evaluma

    rows = [
        {"model": model, "dataset": dataset, "metric": metric, "score": score}
        for model, per_ds in score_map.items()
        for dataset, score in per_ds.items()
    ]
    return evaluma.load_df(
        pd.DataFrame(rows), metric_type_bounds=rl.METRIC_BOUNDS, drop_incomplete=True
    )


# --- Slice 3 tests ---------------------------------------------------------


def test_avg_rank_values():
    pytest.importorskip("evaluma")
    bench = _bench_from_scores(
        {
            "a": {"d1": 0.9, "d2": 0.5},
            "b": {"d1": 0.8, "d2": 0.9},
            "c": {"d1": 0.7, "d2": 0.7},
        }
    )
    ranked = rl.avg_rank(bench)
    by_model = {row["model"]: row["overall"] for row in ranked}
    assert by_model["a"] == pytest.approx(2.0)
    assert by_model["b"] == pytest.approx(1.5)
    assert by_model["c"] == pytest.approx(2.5)
    assert [row["model"] for row in ranked] == ["b", "a", "c"]


def test_avg_rank_baselines_bottom():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    ranked = rl.avg_rank(bench)
    bottom = {ranked[-1]["model"], ranked[-2]["model"]}
    assert bottom == {"rcf", "imagestats"}


# --- Slice 4 tests ---------------------------------------------------------


def test_elo_table_shape():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    ranked = rl.elo(bench, n_bootstrap=200, random_state=123)
    overalls = [row["overall"] for row in ranked]
    assert overalls == sorted(overalls, reverse=True)
    for row in ranked:
        low, high = row["overall_ci"]
        assert low <= row["overall"] <= high


def test_elo_bootstrap_is_seeded():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    a = rl.elo(bench, n_bootstrap=200, random_state=7)
    b = rl.elo(bench, n_bootstrap=200, random_state=7)
    assert a == b


def test_elo_baselines_bottom():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    ranked = rl.elo(bench, n_bootstrap=200, random_state=123)
    bottom = {ranked[-1]["model"], ranked[-2]["model"]}
    assert bottom == {"rcf", "imagestats"}


# --- Slice 5 tests ---------------------------------------------------------


def test_improvability_best_is_zero():
    pytest.importorskip("evaluma")
    bench = _bench_from_scores(
        {
            "a": {"d1": 0.90, "d2": 0.90},
            "b": {"d1": 0.80, "d2": 0.70},
            "c": {"d1": 0.70, "d2": 0.60},
        }
    )
    ranked = rl.improvability(bench)
    by_model = {row["model"]: row["overall"] for row in ranked}
    assert by_model["a"] == pytest.approx(0.0)
    assert ranked[0]["model"] == "a"


def test_improvability_metric_mapping():
    pytest.importorskip("evaluma")
    import evaluma

    frame = pd.DataFrame(
        [
            {"model": "a", "dataset": "d_acc", "metric": "accuracy", "score": 0.9},
            {"model": "a", "dataset": "d_map", "metric": "map", "score": 0.8},
            {"model": "b", "dataset": "d_acc", "metric": "accuracy", "score": 0.7},
            {"model": "b", "dataset": "d_map", "metric": "map", "score": 0.6},
        ]
    )
    bench = evaluma.load_df(frame, metric_type_bounds=rl.METRIC_BOUNDS, drop_incomplete=True)
    ranked = rl.improvability(bench)
    assert {row["model"] for row in ranked} == {"a", "b"}


def test_improvability_baselines_bottom():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert bench is not None
    ranked = rl.improvability(bench)
    bottom = {ranked[-1]["model"], ranked[-2]["model"]}
    assert bottom == {"rcf", "imagestats"}


# --- Slice 6 tests ---------------------------------------------------------


def test_sensitivity_intersection():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench_lin, _ = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    bench_knn, _ = rl.build_slice(harmonized, **{**DEFAULT_SLICE_KEY, "probe": "knn5"})
    assert bench_lin is not None and bench_knn is not None
    exp_models = set(bench_lin.models_) & set(bench_knn.models_)
    exp_datasets = set(bench_lin.datasets_) & set(bench_knn.datasets_)

    sens = rl.axis_sensitivity(
        harmonized,
        DEFAULT_SLICE_KEY,
        axis="probe",
        aggregation="avg_rank",
        axes=_axes(),
    )
    assert sens["n_models"] == len(exp_models)
    assert sens["n_datasets"] == len(exp_datasets)
    assert sens["n_models"] < max(len(bench_lin.models_), len(bench_knn.models_))


def test_sensitivity_tau_range():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    sens = rl.axis_sensitivity(
        harmonized,
        DEFAULT_SLICE_KEY,
        axis="probe",
        aggregation="avg_rank",
        axes=_axes(),
    )
    assert -1.0 <= sens["tau"] <= 1.0
    assert (sens["cond_a"], sens["cond_b"]) == ("linear", "knn5")


def test_sensitivity_identical_flip():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    sens = rl.axis_sensitivity(
        harmonized,
        DEFAULT_SLICE_KEY,
        axis="pooling",
        aggregation="avg_rank",
        axes=_axes(),
    )
    assert sens["tau"] == pytest.approx(1.0)
    assert sens["small_n"] is True


def test_sensitivity_normalization_axis_present():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    sens = rl.axis_sensitivity(
        harmonized,
        DEFAULT_SLICE_KEY,
        axis="normalization",
        aggregation="avg_rank",
        axes=_axes(),
    )
    assert (sens["cond_a"], sens["cond_b"]) == NORMALIZATIONS
    assert sens["tau"] is not None


def test_avg_rank_ranker_matches_regen():
    pytest.importorskip("evaluma")
    bench = _bench_from_scores(
        {
            "a": {"d1": 0.9, "d2": 0.5},
            "b": {"d1": 0.8, "d2": 0.9},
            "c": {"d1": 0.7, "d2": 0.7},
        }
    )
    order_regen = [row["model"] for row in rl.avg_rank(bench)]
    ranker = bench._rank_vector("avg_rank")
    order_evaluma = ranker.sort_values(kind="stable").index.tolist()
    assert order_regen == order_evaluma


# --- Slice 7 tests ---------------------------------------------------------

ROW_KEYS = {
    "model",
    "display",
    "family",
    "is_baseline",
    "overall",
    "overall_ci",
    "n_tasks",
    "per_task",
}


@pytest.fixture(scope="module")
def assembled():
    """Assemble the full ranking cube once (small bootstrap) and share it."""
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    return rl.assemble(harmonized, n_bootstrap=100, random_state=0)


def test_assemble_default_slice_json(assembled):
    rankings, _sensitivity, _meta, _default = assembled
    slice_block = _slice_block(rankings, DEFAULT_SLICE_KEY)
    assert set(slice_block) == {"avg_rank", "elo", "improvability"}
    for agg in ("avg_rank", "elo", "improvability"):
        rows = slice_block[agg]["rows"]
        assert rows
        for row in rows:
            assert set(row) == ROW_KEYS
            assert isinstance(row["per_task"], dict) and row["per_task"]
    json.dumps(rankings, allow_nan=False)


def test_default_slice_baselines_bottom(assembled):
    rankings, _sensitivity, _meta, _default = assembled
    slice_block = _slice_block(rankings, DEFAULT_SLICE_KEY)
    for agg in ("avg_rank", "elo", "improvability"):
        models = [row["model"] for row in slice_block[agg]["rows"]]
        assert {models[-1], models[-2]} == {"rcf", "imagestats"}


def test_sensitivity_tau_bounds_in_output(assembled):
    _rankings, sensitivity, _meta, _default = assembled
    for by_task in sensitivity.values():
        for by_probe in by_task.values():
            for by_band in by_probe.values():
                for by_pool in by_band.values():
                    for by_norm in by_pool.values():
                        for by_axis in by_norm.values():
                            for entry in by_axis.values():
                                if entry["tau"] is not None:
                                    assert -1.0 <= entry["tau"] <= 1.0


def test_sensitivity_has_aggregation_dimension(assembled):
    _rankings, sensitivity, _meta, _default = assembled
    node = sensitivity["classification"]["linear"]["RGB"]["patch-mean"]["bandspec_zscore"]
    assert set(node) == {"avg_rank", "elo", "improvability"}
    for agg_block in node.values():
        assert set(agg_block) == {"probe", "bands", "pooling", "normalization"}


def test_inline_replaces_constants():
    stub = (
        "<script>\n"
        "const RANKINGS = {};\n"
        "const SENSITIVITY = {};\n"
        "const MODEL_META = {};\n"
        "const DEFAULT_SLICE = {};\n"
        "</script>\n"
    )
    blocks = {
        "RANKINGS": {"classification": {"linear": {}}},
        "SENSITIVITY": {"a": 1},
        "MODEL_META": {"rcf": {"display": "Random conv. features"}},
        "DEFAULT_SLICE": {"task": "classification", "aggregation": "avg_rank"},
    }
    out = rl.inline_into_html(stub, blocks)
    for name, obj in blocks.items():
        match = re.search(rf"const {name} = (.*?);", out, re.DOTALL)
        assert match is not None
        assert json.loads(match.group(1)) == obj


# --- Slice 8 tests ---------------------------------------------------------

HTML_PATH = Path(__file__).resolve().parents[1] / "docs" / "_static" / "ranking_explorer.html"


def test_html_has_inline_anchors():
    text = HTML_PATH.read_text()
    for name in ("RANKINGS", "SENSITIVITY", "MODEL_META", "DEFAULT_SLICE"):
        assert re.search(rf"const {name} = .*?;", text, re.DOTALL) is not None
    for control_id in (
        'id="sel-task"',
        'id="sel-probe"',
        'id="sel-bands"',
        'id="sel-pooling"',
        'id="sel-normalization"',
        'id="sel-aggregation"',
        'id="sel-flow-axis"',
        'id="insights"',
        'id="rank-flow-card"',
        'id="rank-dispersion-card"',
        'id="model-detail-card"',
    ):
        assert control_id in text


def test_generator_writes_html(tmp_path):
    import shutil

    pytest.importorskip("evaluma")
    csv_path = tmp_path / "all_results.csv"
    _raw_df().to_csv(csv_path, index=False)
    html_path = tmp_path / "ranking_explorer.html"
    shutil.copy(HTML_PATH, html_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "regen_leaderboard.py"),
            "--csv",
            str(csv_path),
            "--html",
            str(html_path),
            "--bootstrap",
            "50",
            "--seed",
            "13",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    text = html_path.read_text()
    assert text.strip()
    match = re.search(r"const RANKINGS = (.*?);", text, re.DOTALL)
    assert match is not None
    rankings = json.loads(match.group(1))
    assert rankings["classification"]["linear"]["RGB"]["patch-mean"]["bandspec_zscore"]["avg_rank"][
        "rows"
    ]
