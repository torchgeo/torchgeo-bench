"""Post-hoc analysis joining CKA head-row drift with UQ outcome deltas.

Pure pandas/NumPy. Establishes the within-model coupling between corruption-
induced embedding change (``1 - CKA(head)``) and calibration/accuracy
degradation, plus the encoder sensitivity and depth-amplification descriptors.
"""

import numpy as np
import pandas as pd

from torchgeo_bench.cka.metrics import _spearman

_OUTCOME_METRICS = ("nll", "accuracy", "ece")
_JOIN_KEYS = ["name", "dataset", "corruption_type", "severity"]


def build_joined_table(
    cka_df: pd.DataFrame,
    uq_df: pd.DataFrame,
    uq_method: str = "uncalibrated",
) -> pd.DataFrame:
    """Join CKA head-row drift with per-condition UQ outcome deltas.

    Args:
        cka_df: Rows from ``cka_results.csv`` (block + head rows).
        uq_df: Long-format rows from ``uq_focused_results.csv``.
        uq_method: UQ method whose outcomes to join (e.g. ``"uncalibrated"``).

    Returns:
        Long table keyed by ``(name, dataset, corruption_type, severity)`` with
        ``x_drift = 1 - cka(head)`` and ``d_nll``/``d_acc``/``d_ece`` deltas
        relative to the per-``(name, dataset)`` clean baseline. Clean rows are
        dropped.
    """
    heads = cka_df[cka_df["layer_name"] == "head"].copy()
    heads = heads[heads["corruption_type"] != "clean"].copy()
    heads["x_drift"] = 1.0 - heads["cka"].astype(float)

    uq = uq_df[uq_df["uq_method"] == uq_method]
    uq = uq[uq["metric_name"].isin(_OUTCOME_METRICS)]
    wide = uq.pivot_table(
        index=_JOIN_KEYS, columns="metric_name", values="metric_value", aggfunc="mean"
    ).reset_index()

    clean = wide[wide["corruption_type"] == "clean"]
    baseline = clean.set_index(["name", "dataset"])[list(_OUTCOME_METRICS)]
    baseline = baseline.rename(columns={m: f"{m}_clean" for m in _OUTCOME_METRICS})

    corrupted = wide[wide["corruption_type"] != "clean"].merge(
        baseline, on=["name", "dataset"], how="left"
    )
    corrupted["d_nll"] = corrupted["nll"] - corrupted["nll_clean"]
    corrupted["d_acc"] = corrupted["accuracy"] - corrupted["accuracy_clean"]
    corrupted["d_ece"] = corrupted["ece"] - corrupted["ece_clean"]

    joined = heads[_JOIN_KEYS + ["x_drift"]].merge(
        corrupted[_JOIN_KEYS + ["d_nll", "d_acc", "d_ece"]], on=_JOIN_KEYS, how="inner"
    )
    return joined.reset_index(drop=True)


def encoder_sensitivity(cka_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(model, corruption) sensitivity scalar ``1 - mean_{s=1..5} CKA(head)``.

    Args:
        cka_df: Rows from ``cka_results.csv`` (block + head rows).

    Returns:
        One row per ``(name, corruption_type)`` with a ``sensitivity`` column;
        severity-0 clean rows are excluded and corruptions are not averaged
        together.
    """
    heads = cka_df[(cka_df["layer_name"] == "head") & (cka_df["corruption_type"] != "clean")]
    grouped = (
        heads.groupby(["name", "corruption_type"])["cka"].mean().reset_index()
    )
    grouped["sensitivity"] = 1.0 - grouped["cka"].astype(float)
    return grouped[["name", "corruption_type", "sensitivity"]]


def pr_collapse_association(
    cka_df: pd.DataFrame,
    uq_df: pd.DataFrame,
    uq_method: str = "uncalibrated",
) -> pd.DataFrame:
    """Per-model Spearman rho between head PR collapse and calibration delta.

    PR collapse is ``clean_participation_ratio - participation_ratio`` on the
    head (centered PR) row; the calibration delta is ``d_ece`` vs the clean
    baseline. Reported separately from the drift coupling so the PR-collapse
    signal is distinguishable from CKA drift.

    Args:
        cka_df: Rows from ``cka_results.csv`` (block + head rows).
        uq_df: Long-format rows from ``uq_focused_results.csv``.
        uq_method: UQ method whose ECE to use.

    Returns:
        One row per model with ``rho_pr_ece``.
    """
    heads = cka_df[(cka_df["layer_name"] == "head") & (cka_df["corruption_type"] != "clean")].copy()
    heads["pr_drop"] = (
        heads["clean_participation_ratio"].astype(float) - heads["participation_ratio"].astype(float)
    )

    table = build_joined_table(cka_df, uq_df, uq_method=uq_method)
    merged = heads[_JOIN_KEYS + ["pr_drop"]].merge(
        table[_JOIN_KEYS + ["d_ece"]], on=_JOIN_KEYS, how="inner"
    )

    records = []
    for name, g in merged.groupby("name"):
        records.append(
            {
                "name": name,
                "rho_pr_ece": _spearman(
                    g["pr_drop"].to_numpy(dtype=np.float64),
                    g["d_ece"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def depth_amplification(cka_df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Per-(model, severity) depth-amplification ratio and absorber/collapser label.

    The ratio is ``(1 - cka_head) / (1 - cka_shallowest)`` — how the head
    representation amplifies (collapser) or absorbs (absorber) the corruption
    relative to the shallowest block — without assuming a monotone per-layer
    profile. Single-layer encoders (one block) have no depth to amplify across
    and return a NaN ratio.

    Args:
        cka_df: Rows from ``cka_results.csv`` (block + head rows).
        threshold: Ratio cutoff separating absorber (``< threshold``) from
            collapser (``> threshold``).

    Returns:
        One row per ``(name, severity)`` with ``shallow_drift``, ``head_drift``,
        ``amplification_ratio``, ``label`` and the full per-layer ``profile``
        (mean ``1 - cka`` by layer index, head last).
    """
    corrupted = cka_df[cka_df["corruption_type"] != "clean"].copy()
    corrupted["drift"] = 1.0 - corrupted["cka"].astype(float)

    records = []
    for (name, severity), g in corrupted.groupby(["name", "severity"]):
        blocks = g[g["layer_name"] != "head"]
        head = g[g["layer_name"] == "head"]
        profile = g.groupby("layer_index")["drift"].mean().sort_index()

        head_drift = float(head["drift"].mean()) if len(head) else float("nan")
        n_block_layers = int(blocks["layer_index"].nunique())
        if n_block_layers <= 1:
            shallow_drift = float("nan")
            ratio = float("nan")
            label = None
        else:
            shallow_idx = int(blocks["layer_index"].min())
            shallow_drift = float(blocks[blocks["layer_index"] == shallow_idx]["drift"].mean())
            ratio = head_drift / shallow_drift if shallow_drift != 0.0 else float("nan")
            if not np.isfinite(ratio):
                label = None
            elif ratio > threshold:
                label = "collapser"
            elif ratio < threshold:
                label = "absorber"
            else:
                label = "neutral"

        records.append(
            {
                "name": name,
                "severity": severity,
                "shallow_drift": shallow_drift,
                "head_drift": head_drift,
                "amplification_ratio": ratio,
                "label": label,
                "profile": list(profile.to_numpy()),
            }
        )
    return pd.DataFrame.from_records(records)


def within_model_coupling(
    table: pd.DataFrame, fixed_severity: int | None = None
) -> pd.DataFrame:
    """Per-model Spearman rho between head drift and each outcome delta.

    Args:
        table: Output of :func:`build_joined_table`.
        fixed_severity: When given, filter to this severity before correlating
            (the fixed-severity coupling); otherwise pool over all severities.

    Returns:
        One row per model with ``rho_nll``/``rho_acc``/``rho_ece``.
    """
    df = table
    if fixed_severity is not None:
        df = df[df["severity"] == int(fixed_severity)]

    records = []
    for name, g in df.groupby("name"):
        x = g["x_drift"].to_numpy(dtype=np.float64)
        records.append(
            {
                "name": name,
                "rho_nll": _spearman(x, g["d_nll"].to_numpy(dtype=np.float64)),
                "rho_acc": _spearman(x, g["d_acc"].to_numpy(dtype=np.float64)),
                "rho_ece": _spearman(x, g["d_ece"].to_numpy(dtype=np.float64)),
            }
        )
    return pd.DataFrame.from_records(records)
