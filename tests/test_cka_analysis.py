import numpy as np
import pandas as pd

from torchgeo_bench.cka.analysis import (
    build_joined_table,
    depth_amplification,
    encoder_sensitivity,
    pr_collapse_association,
    within_model_coupling,
)


def _cka_head_row(name, dataset, corruption, severity, cka, layer_name="head"):  # noqa: ANN001
    return {
        "name": name,
        "dataset": dataset,
        "corruption_type": corruption,
        "severity": severity,
        "layer_name": layer_name,
        "layer_index": 4 if layer_name == "head" else 0,
        "cka": cka,
    }


def _uq_rows(name, dataset, corruption, severity, nll, acc, ece, uq_method="uncalibrated"):  # noqa: ANN001
    base = {
        "name": name,
        "dataset": dataset,
        "uq_method": uq_method,
        "corruption_type": corruption,
        "severity": severity,
    }
    return [
        {**base, "metric_name": "nll", "metric_value": nll},
        {**base, "metric_name": "accuracy", "metric_value": acc},
        {**base, "metric_name": "ece", "metric_value": ece},
    ]


def test_build_joined_table():
    cka_df = pd.DataFrame(
        [
            _cka_head_row("A", "d1", "clean", 0, 1.0),
            _cka_head_row("A", "d1", "c1", 1, 0.9),
            _cka_head_row("A", "d1", "c1", 2, 0.8),
            # a non-head block row that must be filtered out
            _cka_head_row("A", "d1", "c1", 1, 0.5, layer_name="backbone.layer1"),
        ]
    )
    uq_rows = (
        _uq_rows("A", "d1", "clean", 0, nll=0.5, acc=0.90, ece=0.05)
        + _uq_rows("A", "d1", "c1", 1, nll=0.7, acc=0.85, ece=0.08)
        + _uq_rows("A", "d1", "c1", 2, nll=0.9, acc=0.80, ece=0.12)
    )
    uq_df = pd.DataFrame(uq_rows)

    table = build_joined_table(cka_df, uq_df, uq_method="uncalibrated")

    for col in ("name", "dataset", "corruption_type", "severity", "x_drift", "d_nll", "d_acc", "d_ece"):
        assert col in table.columns
    assert (table["corruption_type"] != "clean").all()
    assert len(table) == 2

    s1 = table[table["severity"] == 1].iloc[0]
    assert np.isclose(s1["x_drift"], 0.1)
    assert np.isclose(s1["d_nll"], 0.2)
    assert np.isclose(s1["d_acc"], -0.05)
    assert np.isclose(s1["d_ece"], 0.03)

    s2 = table[table["severity"] == 2].iloc[0]
    assert np.isclose(s2["x_drift"], 0.2)
    assert np.isclose(s2["d_nll"], 0.4)
    assert np.isclose(s2["d_acc"], -0.10)
    assert np.isclose(s2["d_ece"], 0.07)


def test_within_model_coupling_perfect_monotone():
    table = pd.DataFrame(
        {
            "name": ["A", "A", "A", "A"],
            "dataset": ["d1"] * 4,
            "corruption_type": ["c1"] * 4,
            "severity": [1, 2, 3, 4],
            "x_drift": [0.1, 0.2, 0.3, 0.4],
            "d_nll": [1.0, 2.0, 3.0, 4.0],  # strictly increasing in x_drift
            "d_acc": [-0.1, -0.2, -0.3, -0.4],
            "d_ece": [0.01, 0.02, 0.03, 0.04],
        }
    )
    out = within_model_coupling(table)
    assert len(out) == 1  # one rho per model
    row = out[out["name"] == "A"].iloc[0]
    assert np.isclose(row["rho_nll"], 1.0)
    assert np.isclose(row["rho_ece"], 1.0)


def test_within_model_coupling_fixed_severity():
    table = pd.DataFrame(
        {
            "name": ["A", "A", "A", "A"],
            "dataset": ["d1"] * 4,
            "corruption_type": ["c1"] * 4,
            "severity": [4, 4, 5, 5],
            # At severity 5: x_drift and d_nll are perfectly monotone (rho=1).
            # The severity-4 rows would break monotonicity if not filtered.
            "x_drift": [0.9, 0.1, 0.2, 0.4],
            "d_nll": [0.0, 5.0, 1.0, 2.0],
            "d_acc": [0.0, 0.0, -0.1, -0.2],
            "d_ece": [0.0, 0.0, 0.01, 0.02],
        }
    )
    out = within_model_coupling(table, fixed_severity=5)
    assert len(out) == 1  # one rho per model
    row = out[out["name"] == "A"].iloc[0]
    assert np.isclose(row["rho_nll"], 1.0)


def _head_severity_rows(name, dataset, corruption, ckas):  # noqa: ANN001
    return [
        _cka_head_row(name, dataset, corruption, sev, cka)
        for sev, cka in zip(range(1, 6), ckas, strict=True)
    ]


def test_encoder_sensitivity_per_corruption():
    rows = [_cka_head_row("A", "d1", "clean", 0, 1.0)]  # ignored
    rows += _head_severity_rows("A", "d1", "c1", [0.9, 0.8, 0.7, 0.6, 0.5])  # mean 0.7
    rows += _head_severity_rows("A", "d1", "c2", [0.5, 0.5, 0.5, 0.5, 0.5])  # mean 0.5
    # a non-head block row that must be ignored
    rows.append(_cka_head_row("A", "d1", "c1", 1, 0.1, layer_name="backbone.layer1"))
    cka_df = pd.DataFrame(rows)

    out = encoder_sensitivity(cka_df)
    c1 = out[(out["name"] == "A") & (out["corruption_type"] == "c1")].iloc[0]
    c2 = out[(out["name"] == "A") & (out["corruption_type"] == "c2")].iloc[0]
    assert np.isclose(c1["sensitivity"], 0.3)  # 1 - 0.7
    assert np.isclose(c2["sensitivity"], 0.5)  # 1 - 0.5, not cross-corruption averaged


def test_encoder_sensitivity_shape():
    rows = _head_severity_rows("A", "d1", "c1", [0.9] * 5)
    rows += _head_severity_rows("A", "d1", "c2", [0.8] * 5)
    cka_df = pd.DataFrame(rows)
    out = encoder_sensitivity(cka_df)
    assert len(out) == 2  # one row per (model, corruption)
    assert set(map(tuple, out[["name", "corruption_type"]].to_numpy())) == {("A", "c1"), ("A", "c2")}


def _cka_layer_row(name, dataset, corruption, severity, cka, layer_name, layer_index):  # noqa: ANN001
    return {
        "name": name,
        "dataset": dataset,
        "corruption_type": corruption,
        "severity": severity,
        "layer_name": layer_name,
        "layer_index": layer_index,
        "cka": cka,
    }


def test_depth_amplification_ratio():
    rows = [
        # Collapser C: shallow drift 0.05, head drift 0.5 -> ratio 10 (> 1).
        _cka_layer_row("C", "d1", "c1", 3, 0.95, "backbone.layer1", 0),
        _cka_layer_row("C", "d1", "c1", 3, 0.90, "backbone.layer2", 1),
        _cka_layer_row("C", "d1", "c1", 3, 0.50, "head", 2),
        # Absorber B: shallow drift 0.5, head drift 0.05 -> ratio 0.1 (< 1).
        _cka_layer_row("B", "d1", "c1", 3, 0.50, "backbone.layer1", 0),
        _cka_layer_row("B", "d1", "c1", 3, 0.70, "backbone.layer2", 1),
        _cka_layer_row("B", "d1", "c1", 3, 0.95, "head", 2),
    ]
    out = depth_amplification(pd.DataFrame(rows), threshold=1.0)
    c = out[(out["name"] == "C") & (out["severity"] == 3)].iloc[0]
    b = out[(out["name"] == "B") & (out["severity"] == 3)].iloc[0]
    assert c["amplification_ratio"] > 1.0
    assert c["label"] == "collapser"
    assert b["amplification_ratio"] < 1.0
    assert b["label"] == "absorber"


def test_depth_amplification_handles_single_layer():
    rows = [
        _cka_layer_row("R", "d1", "c1", 1, 0.8, "rcf", 0),
        _cka_layer_row("R", "d1", "c1", 1, 0.7, "head", 1),
    ]
    out = depth_amplification(pd.DataFrame(rows))
    r = out[out["name"] == "R"].iloc[0]
    assert np.isnan(r["amplification_ratio"])


def test_depth_amplification_non_monotone_profile():
    rows = [
        _cka_layer_row("M", "d1", "c1", 2, 0.9, "backbone.layer1", 0),  # shallow drift 0.1
        _cka_layer_row("M", "d1", "c1", 2, 0.3, "backbone.layer2", 1),  # mid peak drift 0.7
        _cka_layer_row("M", "d1", "c1", 2, 0.6, "backbone.layer3", 2),
        _cka_layer_row("M", "d1", "c1", 2, 0.8, "head", 3),  # head drift 0.2
    ]
    out = depth_amplification(pd.DataFrame(rows))
    m = out[out["name"] == "M"].iloc[0]
    # ratio uses head vs shallowest only, ignoring the mid-network peak.
    assert np.isclose(m["amplification_ratio"], 2.0)
    # full per-layer profile preserved (4 entries: layers 0,1,2 + head).
    assert len(m["profile"]) == 4


def _pr_head_row(name, dataset, corruption, severity, pr, clean_pr, cka=0.9):  # noqa: ANN001
    return {
        "name": name,
        "dataset": dataset,
        "corruption_type": corruption,
        "severity": severity,
        "layer_name": "head",
        "layer_index": 4,
        "cka": cka,
        "participation_ratio": pr,
        "clean_participation_ratio": clean_pr,
    }


def test_pr_collapse_association():
    cka_rows = [
        _pr_head_row("A", "d1", "clean", 0, 10.0, 10.0, cka=1.0),
        _pr_head_row("A", "d1", "c1", 1, 9.0, 10.0),  # pr_drop 1
        _pr_head_row("A", "d1", "c1", 2, 8.0, 10.0),  # pr_drop 2
        _pr_head_row("A", "d1", "c1", 3, 7.0, 10.0),  # pr_drop 3
    ]
    uq_rows = (
        _uq_rows("A", "d1", "clean", 0, nll=0.5, acc=0.9, ece=0.05)
        + _uq_rows("A", "d1", "c1", 1, nll=0.6, acc=0.88, ece=0.06)  # d_ece 0.01
        + _uq_rows("A", "d1", "c1", 2, nll=0.7, acc=0.85, ece=0.08)  # d_ece 0.03
        + _uq_rows("A", "d1", "c1", 3, nll=0.8, acc=0.80, ece=0.11)  # d_ece 0.06
    )
    out = pr_collapse_association(pd.DataFrame(cka_rows), pd.DataFrame(uq_rows))
    row = out[out["name"] == "A"].iloc[0]
    assert np.isclose(row["rho_pr_ece"], 1.0)


def test_pr_collapse_uses_centered_pr():
    # participation_ratio is constant; the PR collapse comes entirely from the
    # clean_participation_ratio baseline. A correct implementation reads both
    # columns (drop = clean_pr - pr) and yields a finite, monotone rho; using
    # the raw (constant) participation_ratio alone would give NaN.
    cka_rows = [
        _pr_head_row("A", "d1", "clean", 0, 5.0, 5.0, cka=1.0),
        _pr_head_row("A", "d1", "c1", 1, 5.0, 6.0),  # pr_drop 1
        _pr_head_row("A", "d1", "c1", 2, 5.0, 7.0),  # pr_drop 2
        _pr_head_row("A", "d1", "c1", 3, 5.0, 8.0),  # pr_drop 3
    ]
    uq_rows = (
        _uq_rows("A", "d1", "clean", 0, nll=0.5, acc=0.9, ece=0.05)
        + _uq_rows("A", "d1", "c1", 1, nll=0.6, acc=0.88, ece=0.06)
        + _uq_rows("A", "d1", "c1", 2, nll=0.7, acc=0.85, ece=0.07)
        + _uq_rows("A", "d1", "c1", 3, nll=0.8, acc=0.80, ece=0.08)
    )
    out = pr_collapse_association(pd.DataFrame(cka_rows), pd.DataFrame(uq_rows))
    row = out[out["name"] == "A"].iloc[0]
    assert np.isfinite(row["rho_pr_ece"])
    assert np.isclose(row["rho_pr_ece"], 1.0)
