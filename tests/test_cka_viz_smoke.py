import subprocess
import sys

import pandas as pd
import pytest


def _cka_rows():
    rows = []
    for name in ("A", "B"):
        # clean baseline rows (block + head)
        rows.append(_row(name, "clean", 0, 0, "backbone.layer1", 1.0, pr=10.0, clean_pr=10.0))
        rows.append(_row(name, "clean", 0, 4, "head", 1.0, pr=10.0, clean_pr=10.0))
        for sev in (1, 2, 3):
            block_cka = 1.0 - 0.05 * sev
            head_cka = 1.0 - 0.1 * sev * (1.0 if name == "A" else 1.5)
            rows.append(_row(name, "c1", sev, 0, "backbone.layer1", block_cka, pr=10.0, clean_pr=10.0))
            rows.append(_row(name, "c1", sev, 4, "head", head_cka, pr=10.0 - sev, clean_pr=10.0))
    return rows


def _row(name, corruption, severity, layer_index, layer_name, cka, pr, clean_pr):  # noqa: ANN001
    return {
        "name": name,
        "dataset": "d1",
        "corruption_type": corruption,
        "severity": severity,
        "layer_index": layer_index,
        "layer_name": layer_name,
        "cka": cka,
        "participation_ratio": pr,
        "clean_participation_ratio": clean_pr,
    }


def _uq_rows():
    rows = []
    for name in ("A", "B"):
        for corr, sev, nll, acc, ece in [
            ("clean", 0, 0.50, 0.90, 0.05),
            ("c1", 1, 0.60, 0.88, 0.06),
            ("c1", 2, 0.70, 0.85, 0.08),
            ("c1", 3, 0.85, 0.80, 0.11),
        ]:
            base = {
                "name": name,
                "dataset": "d1",
                "uq_method": "uncalibrated",
                "corruption_type": corr,
                "severity": sev,
            }
            rows.append({**base, "metric_name": "nll", "metric_value": nll})
            rows.append({**base, "metric_name": "accuracy", "metric_value": acc})
            rows.append({**base, "metric_name": "ece", "metric_value": ece})
    return rows


def test_cka_figures_smoke(tmp_path):
    pytest.importorskip("matplotlib")

    cka_csv = tmp_path / "cka_results.csv"
    uq_csv = tmp_path / "uq_focused_results.csv"
    pd.DataFrame(_cka_rows()).to_csv(cka_csv, index=False)
    pd.DataFrame(_uq_rows()).to_csv(uq_csv, index=False)

    outdir = tmp_path / "figs"
    cmd = [
        sys.executable,
        "viz/cka_drift_sensitivity_figures.py",
        "--cka-csv",
        str(cka_csv),
        "--uq-csv",
        str(uq_csv),
        "--outdir",
        str(outdir),
        "--format",
        "png",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr

    expected = [
        "cka_headline_scatter.png",
        "cka_rho_summary.png",
        "cka_depth_profiles.png",
        "cka_sensitivity.png",
    ]
    for fname in expected:
        fpath = outdir / fname
        assert fpath.exists(), f"missing {fname}: {result.stderr}"
        assert fpath.stat().st_size > 0
