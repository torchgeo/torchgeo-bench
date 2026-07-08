"""Smoke + unit tests for ``experiments/scripts/regen_leaderboard.py``.

The generator is a standalone script (not an installed module), so we import it
by inserting ``experiments/scripts`` on ``sys.path`` — mirroring the convention
in ``test_geofm_cka_prototypes_smoke.py``. Tests that need evaluma skip
gracefully when it is not installed.
"""

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

SINGLE_LABEL = ["m-eurosat", "m-so2sat", "m-pv4ger"]  # metric accuracy
MULTI_LABEL = ["m-bigearthnet", "treesatai"]  # metric micro_mAP
DATASETS = SINGLE_LABEL + MULTI_LABEL

# base quality per model; higher = better. baselines sit at the bottom.
BASE_SCORE = {"good": 0.90, "mid": 0.80, "low": 0.70, "rcf": 0.45, "imagestats": 0.40}
CLS_MODELS = {"good", "mid", "low"}
MS_BANDS_A = "blue,green,red,nir,swir_1,swir_2"
MS_BANDS_B = "b02,b03,b04,b08,b05,b06"


def _metric_for(dataset: str) -> str:
    return "accuracy" if dataset in SINGLE_LABEL else "micro_mAP"


def _score(base: str, dataset: str, probe: str, bandclass: str, pooling: str) -> float:
    s = BASE_SCORE[base]
    s += 0.0 if probe == "linear" else -0.03
    s += 0.02 if bandclass == "Multispectral" else 0.0
    s += -0.01 if pooling == "cls-token" else 0.0
    s += 0.001 * DATASETS.index(dataset)  # deterministic per-dataset jitter
    return round(s, 4)


def _row(name: str, dataset: str, probe: str, bands: str, score: float) -> dict[str, object]:
    return {
        "name": name,
        "dataset": dataset,
        "method": probe,
        "metric_name": _metric_for(dataset),
        "metric_value": score,
        "bands": bands,
    }


def _raw_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for base in BASE_SCORE:
        for dataset in DATASETS:
            for probe in ("linear", "knn5"):
                # patch-mean RGB + Multispectral
                rows.append(
                    _row(
                        base,
                        dataset,
                        probe,
                        "rgb",
                        _score(base, dataset, probe, "RGB", "patch-mean"),
                    )
                )
                rows.append(
                    _row(
                        base,
                        dataset,
                        probe,
                        MS_BANDS_A,
                        _score(base, dataset, probe, "Multispectral", "patch-mean"),
                    )
                )
                # cls-token readout for dual-readout backbones
                if base in CLS_MODELS:
                    rows.append(
                        _row(
                            f"{base}_cls",
                            dataset,
                            probe,
                            "rgb",
                            _score(base, dataset, probe, "RGB", "cls-token"),
                        )
                    )
                    rows.append(
                        _row(
                            f"{base}_cls",
                            dataset,
                            probe,
                            MS_BANDS_A,
                            _score(base, dataset, probe, "Multispectral", "cls-token"),
                        )
                    )

    # a second Multispectral band-list row for one cell (max-collapse fixture):
    # higher score, should win the collapse.
    rows.append(_row("good", "m-eurosat", "linear", MS_BANDS_B, 0.99))

    # a probe-asymmetric model: complete under RGB/linear/patch-mean but with no
    # knn5 rows, so the probe-axis intersection is strictly smaller than the
    # linear roster.
    for dataset in DATASETS:
        rows.append(
            _row(
                "lin_only",
                dataset,
                "linear",
                "rgb",
                _score("mid", dataset, "linear", "RGB", "patch-mean"),
            )
        )

    # OlmoEarth-style single-dataset ablation rows — must be dropped.
    for suffix in ("model_lsat_x100", "model_naip_rgb", "model_sar_db"):
        rows.append(_row(suffix, "m-eurosat", "linear", "rgb", 0.5))

    # a partial-coverage model: present in every RGB linear patch-mean dataset
    # except one — should be excluded from that slice, not seated.
    for dataset in DATASETS:
        if dataset == "treesatai":
            continue
        rows.append(
            _row(
                "partial",
                dataset,
                "linear",
                "rgb",
                _score("mid", dataset, "linear", "RGB", "patch-mean"),
            )
        )

    return rows


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(_raw_rows())


# --- Slice 1 tests ---------------------------------------------------------


def test_drops_ablation_rows():
    out = rl.harmonize(_raw_df())
    for tag in ("_lsat_", "_naip_", "_sar_"):
        assert not out["base_model"].astype(str).str.contains(tag).any()


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
    # only bands == "rgb" maps to RGB
    rgb = out[out["bandclass"] == "RGB"]
    assert not rgb.empty


def test_canonical_metric_map():
    out = rl.harmonize(_raw_df())
    assert set(out["metric"].unique()) == {"accuracy", "map"}


def test_max_score_per_cell():
    out = rl.harmonize(_raw_df())
    cell = out[
        (out["base_model"] == "good")
        & (out["dataset"] == "m-eurosat")
        & (out["probe"] == "linear")
        & (out["bandclass"] == "Multispectral")
        & (out["pooling"] == "patch-mean")
    ]
    assert len(cell) == 1
    assert cell["score"].iloc[0] == pytest.approx(0.99)


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
    assert not any("_lsat_" in m for m in params)
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
    _r, _s, meta, _d = rl.assemble(harmonized, n_bootstrap=50, params=params)
    assert meta["good"]["params_m"] == pytest.approx(304.3)
    # a seated model with no recorded params row stays None (renders as em-dash)
    assert meta["low"]["params_m"] is None


# --- Slice 2 tests ---------------------------------------------------------

DEFAULT_SLICE_KEY = {
    "task": "classification",
    "probe": "linear",
    "bandclass": "RGB",
    "pooling": "patch-mean",
}


def test_build_slice_complete_matrix():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    scores = bench.scores_
    assert not scores.isna().to_numpy().any()
    assert set(bench.datasets_) == set(DATASETS)


def test_build_slice_records_excluded():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert "partial" not in set(bench.models_)
    entry = next(e for e in excluded if e["model"] == "partial")
    assert entry["n_tasks"] == len(DATASETS) - 1


def test_build_slice_keeps_baselines():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    assert {"rcf", "imagestats"}.issubset(set(bench.models_))


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
    # d1 ranks: a=1,b=2,c=3 ; d2 ranks: b=1,c=2,a=3 -> means a=2.0,b=1.5,c=2.5
    by_model = {r["model"]: r["overall"] for r in ranked}
    assert by_model["a"] == pytest.approx(2.0)
    assert by_model["b"] == pytest.approx(1.5)
    assert by_model["c"] == pytest.approx(2.5)
    assert [r["model"] for r in ranked] == ["b", "a", "c"]  # ascending, best first


def test_avg_rank_baselines_bottom():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    ranked = rl.avg_rank(bench)
    bottom = {ranked[-1]["model"], ranked[-2]["model"]}
    assert bottom == {"rcf", "imagestats"}


# --- Slice 4 tests ---------------------------------------------------------


def test_elo_table_shape():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    ranked = rl.elo(bench, n_bootstrap=200)
    overalls = [r["overall"] for r in ranked]
    assert overalls == sorted(overalls, reverse=True)  # ELO descending
    for r in ranked:
        low, high = r["overall_ci"]
        assert low <= r["overall"] <= high


def test_elo_baselines_bottom():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    ranked = rl.elo(bench, n_bootstrap=200)
    bottom = {ranked[-1]["model"], ranked[-2]["model"]}
    assert bottom == {"rcf", "imagestats"}


# --- Slice 5 tests ---------------------------------------------------------


def test_improvability_best_is_zero():
    pytest.importorskip("evaluma")
    bench = _bench_from_scores(
        {
            "a": {"d1": 0.90, "d2": 0.90},  # best on both datasets
            "b": {"d1": 0.80, "d2": 0.70},
            "c": {"d1": 0.70, "d2": 0.60},
        }
    )
    ranked = rl.improvability(bench)
    by_model = {r["model"]: r["overall"] for r in ranked}
    assert by_model["a"] == pytest.approx(0.0)
    assert ranked[0]["model"] == "a"  # ascending, 0 = best


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
    ranked = rl.improvability(bench)  # must not raise on the mixed metrics
    assert {r["model"] for r in ranked} == {"a", "b"}


def test_improvability_baselines_bottom():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench, _excluded = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    ranked = rl.improvability(bench)
    bottom = {ranked[-1]["model"], ranked[-2]["model"]}
    assert bottom == {"rcf", "imagestats"}


# --- Slice 6 tests ---------------------------------------------------------


def test_sensitivity_intersection():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    bench_lin, _ = rl.build_slice(harmonized, **DEFAULT_SLICE_KEY)
    bench_knn, _ = rl.build_slice(harmonized, **{**DEFAULT_SLICE_KEY, "probe": "knn5"})
    exp_models = set(bench_lin.models_) & set(bench_knn.models_)
    exp_datasets = set(bench_lin.datasets_) & set(bench_knn.datasets_)

    sens = rl.axis_sensitivity(harmonized, DEFAULT_SLICE_KEY, axis="probe", aggregation="avg_rank")
    assert sens["n_models"] == len(exp_models)
    assert sens["n_datasets"] == len(exp_datasets)
    # intersection is strictly smaller than the (larger) linear roster
    assert sens["n_models"] < max(len(bench_lin.models_), len(bench_knn.models_))


def test_sensitivity_tau_range():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    sens = rl.axis_sensitivity(harmonized, DEFAULT_SLICE_KEY, axis="probe", aggregation="avg_rank")
    assert -1.0 <= sens["tau"] <= 1.0
    assert (sens["cond_a"], sens["cond_b"]) == ("linear", "knn5")


def test_sensitivity_identical_flip():
    pytest.importorskip("evaluma")
    harmonized = rl.harmonize(_raw_df())
    # pooling flip on the RGB slice: patch-mean vs cls-token share good/mid/low
    # with the same ordering -> tau == 1.0 on a tiny (small-N) intersection.
    sens = rl.axis_sensitivity(
        harmonized, DEFAULT_SLICE_KEY, axis="pooling", aggregation="avg_rank"
    )
    assert sens["tau"] == pytest.approx(1.0)
    assert sens["small_n"] is True


def test_avg_rank_ranker_matches_regen():
    # Open-Questions guard: evaluma's "avg_rank" ranker (drives SENSITIVITY) and
    # the script's own avg_rank (drives RANKINGS rows) must agree on ordering, or
    # the tau panel and the table tell different stories. Tie-free fixture so the
    # ordering is unambiguous.
    pytest.importorskip("evaluma")
    bench = _bench_from_scores(
        {
            "a": {"d1": 0.9, "d2": 0.5},
            "b": {"d1": 0.8, "d2": 0.9},
            "c": {"d1": 0.7, "d2": 0.7},
        }
    )
    order_regen = [r["model"] for r in rl.avg_rank(bench)]
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
    return rl.assemble(harmonized, n_bootstrap=100)


def test_assemble_default_slice_json(assembled):
    import json

    rankings, _sensitivity, _meta, _default = assembled
    slice_block = rankings["classification"]["linear"]["RGB"]["patch-mean"]
    assert set(slice_block) == {"avg_rank", "elo", "improvability"}
    for agg in ("avg_rank", "elo", "improvability"):
        rows = slice_block[agg]["rows"]
        assert rows
        for row in rows:
            assert set(row) == ROW_KEYS
            assert isinstance(row["per_task"], dict) and row["per_task"]
    # valid JSON, no NaN/inf
    json.dumps(rankings, allow_nan=False)


def test_default_slice_baselines_bottom(assembled):
    rankings, _sensitivity, _meta, _default = assembled
    slice_block = rankings["classification"]["linear"]["RGB"]["patch-mean"]
    for agg in ("avg_rank", "elo", "improvability"):
        models = [r["model"] for r in slice_block[agg]["rows"]]
        assert {models[-1], models[-2]} == {"rcf", "imagestats"}


def test_sensitivity_tau_bounds_in_output(assembled):
    _rankings, sensitivity, _meta, _default = assembled
    for by_probe in sensitivity.values():
        for by_band in by_probe.values():
            for by_pool in by_band.values():
                for by_agg in by_pool.values():
                    for axis_block in by_agg.values():
                        for entry in axis_block.values():
                            if entry["tau"] is not None:
                                assert -1.0 <= entry["tau"] <= 1.0


def test_sensitivity_has_aggregation_dimension(assembled):
    _rankings, sensitivity, _meta, _default = assembled
    node = sensitivity["classification"]["linear"]["RGB"]["patch-mean"]
    assert set(node) == {"avg_rank", "elo", "improvability"}
    for agg_block in node.values():
        assert set(agg_block) == {"probe", "bands", "pooling"}


def test_inline_replaces_constants():
    import json

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
        m = re.search(rf"const {name} = (.*?);", out, re.DOTALL)
        assert m is not None
        assert json.loads(m.group(1)) == obj


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
        'id="sel-aggregation"',
        'id="sel-flow-axis"',
        'id="insights"',
        'id="rank-flow-card"',
        'id="rank-dispersion-card"',
        'id="model-detail-card"',
    ):
        assert control_id in text


def test_generator_writes_html(tmp_path):
    import json
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
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    text = html_path.read_text()
    assert text.strip()
    m = re.search(r"const RANKINGS = (.*?);", text, re.DOTALL)
    assert m is not None
    rankings = json.loads(m.group(1))
    assert rankings["classification"]["linear"]["RGB"]["patch-mean"]["avg_rank"]["rows"]
