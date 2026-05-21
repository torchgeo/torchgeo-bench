#!/usr/bin/env python
"""Generate exploratory UQ result figures and Tier 1 diagnostics from a long-form CSV."""

import argparse
import logging
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from torchgeo_bench.uq.reliability import build_reliability_frame
from torchgeo_bench.uq.traces import scan_traces

logger = logging.getLogger(__name__)

PROBABILISTIC_METHODS: tuple[str, ...] = (
    "uncalibrated",
    "temp_scaling",
    "laplace",
    "deep_ensemble",
    "bootstrap_ensemble",
)
CONFORMAL_METHOD = "conformal"
ALL_METHODS: tuple[str, ...] = PROBABILISTIC_METHODS + (CONFORMAL_METHOD,)

PROBABILISTIC_CORE_METRICS: tuple[str, ...] = (
    "accuracy",
    "ece",
    "nll",
    "brier",
    "selective_acc_90",
)
PROBABILISTIC_SECONDARY_METRICS: tuple[str, ...] = (
    "predictive_entropy",
    "eaurc",
    "raw_aurc",
)
PROBABILISTIC_ALL_METRICS: tuple[str, ...] = PROBABILISTIC_CORE_METRICS + PROBABILISTIC_SECONDARY_METRICS
CONFORMAL_METRICS: tuple[str, ...] = (
    "accuracy",
    "empirical_coverage",
    "mean_set_size",
)

PROB_METHOD_COLORS: dict[str, str] = {
    "uncalibrated": "#1f77b4",
    "temp_scaling": "#ff7f0e",
    "laplace": "#2ca02c",
    "deep_ensemble": "#d62728",
    "bootstrap_ensemble": "#9467bd",
}
PROB_METHOD_MARKERS: dict[str, str] = {
    "uncalibrated": "o",
    "temp_scaling": "s",
    "laplace": "^",
    "deep_ensemble": "D",
    "bootstrap_ensemble": "P",
}
CONFORMAL_COLOR = "#6c757d"

REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset",
    "uq_method",
    "corruption_type",
    "severity",
    "metric_name",
    "metric_value",
)

DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "alert_type",
    "row_index",
    "dataset",
    "backbone",
    "uq_method",
    "corruption_type",
    "severity",
    "metric_name",
    "value",
    "message",
)


def _import_plotting():
    """Import optional plotting dependency and return pyplot."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plot_uq_results. Install `torchgeo-bench[viz]` "
            "or run `uv sync --extra viz`."
        ) from exc
    return plt


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Path to long-form UQ results CSV.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory root.")
    parser.add_argument("--backbones", type=str, default=None, help="Comma-separated backbone names.")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names.")
    parser.add_argument(
        "--corruptions",
        type=str,
        default=None,
        help="Comma-separated corruption names for trend plots (e.g. cloud,poisson_gaussian).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric names. Defaults to core probabilistic + conformal metrics.",
    )
    parser.add_argument(
        "--calibration-metric",
        type=str,
        default="ece",
        help=(
            "Metric for per-severity probabilistic calibration summary plots "
            "(one line per backbone/model). Set to empty string to disable."
        ),
    )
    parser.add_argument("--format", type=str, default="png", help="Output format (v1 supports png).")
    parser.add_argument("--dpi", type=int, default=200, help="Raster DPI.")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=None,
        help="Optional trace parquet dataset root for reliability plots.",
    )
    parser.add_argument(
        "--reliability-bins",
        type=int,
        default=15,
        help="Number of reliability bins when using traces.",
    )
    parser.add_argument(
        "--reliability-binning",
        type=str,
        default="equal_width",
        help="Reliability binning strategy: equal_width or equal_mass.",
    )
    parser.add_argument(
        "--reliability-use-cache",
        action="store_true",
        help="Use <trace-dir>/reliability_cache.parquet when available.",
    )
    parser.add_argument(
        "--reliability-outdir",
        type=Path,
        default=None,
        help="Optional output directory for reliability plots.",
    )
    parser.add_argument(
        "--no-confidence-trends",
        action="store_true",
        default=False,
        help="Disable the accuracy-vs-confidence overlay trend plots.",
    )
    return parser


def _append_alert(
    alerts: list[dict[str, object]],
    alert_type: str,
    *,
    row_index: int | None = None,
    dataset: str | None = None,
    backbone: str | None = None,
    uq_method: str | None = None,
    corruption_type: str | None = None,
    severity: int | str | None = None,
    metric_name: str | None = None,
    value: str | float | int | None = None,
    message: str = "",
) -> None:
    alerts.append(
        {
            "alert_type": alert_type,
            "row_index": row_index,
            "dataset": dataset,
            "backbone": backbone,
            "uq_method": uq_method,
            "corruption_type": corruption_type,
            "severity": severity,
            "metric_name": metric_name,
            "value": value,
            "message": message,
        }
    )


def _resolve_backbone(df: pd.DataFrame) -> pd.Series:
    if "backbone" in df.columns:
        return df["backbone"].fillna("").astype("string")
    if "name" in df.columns:
        return df["name"].fillna("").astype("string")
    if "model" in df.columns:
        return df["model"].fillna("").astype("string")
    return pd.Series(["unknown"] * len(df), index=df.index, dtype="string")


def _resolve_model_name(df: pd.DataFrame) -> pd.Series:
    if "name" in df.columns:
        return df["name"].fillna("").astype("string")
    if "backbone" in df.columns:
        return df["backbone"].fillna("").astype("string")
    if "model" in df.columns:
        return df["model"].fillna("").astype("string")
    return pd.Series(["unknown"] * len(df), index=df.index, dtype="string")


def _normalize_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    alerts: list[dict[str, object]] = []
    normalized = df.copy()

    for col in REQUIRED_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = pd.NA
            _append_alert(
                alerts,
                "missing_required_column",
                metric_name=col,
                message=f"Missing required column '{col}' in input CSV.",
            )

    normalized["dataset"] = normalized["dataset"].fillna("").astype("string")
    normalized["uq_method"] = normalized["uq_method"].fillna("").astype("string").str.lower()
    normalized["corruption_type"] = normalized["corruption_type"].fillna("").astype("string").str.lower()
    normalized["metric_name"] = normalized["metric_name"].fillna("").astype("string").str.lower()
    normalized["backbone"] = _resolve_backbone(normalized).fillna("").astype("string")
    normalized.loc[normalized["backbone"].str.strip() == "", "backbone"] = "unknown"
    normalized["model_name"] = _resolve_model_name(normalized).fillna("").astype("string")
    normalized.loc[normalized["model_name"].str.strip() == "", "model_name"] = "unknown"

    severity_num = pd.to_numeric(normalized["severity"], errors="coerce")
    invalid_numeric = severity_num.isna() & normalized["severity"].notna()
    for idx in normalized.index[invalid_numeric]:
        _append_alert(
            alerts,
            "invalid_severity",
            row_index=int(idx),
            dataset=str(normalized.at[idx, "dataset"]),
            backbone=str(normalized.at[idx, "backbone"]),
            uq_method=str(normalized.at[idx, "uq_method"]),
            corruption_type=str(normalized.at[idx, "corruption_type"]),
            metric_name=str(normalized.at[idx, "metric_name"]),
            value=str(normalized.at[idx, "severity"]),
            message="Severity is not numeric.",
        )

    is_non_integer = severity_num.notna() & ~np.isclose(severity_num, np.round(severity_num))
    for idx in normalized.index[is_non_integer]:
        _append_alert(
            alerts,
            "invalid_severity",
            row_index=int(idx),
            dataset=str(normalized.at[idx, "dataset"]),
            backbone=str(normalized.at[idx, "backbone"]),
            uq_method=str(normalized.at[idx, "uq_method"]),
            corruption_type=str(normalized.at[idx, "corruption_type"]),
            metric_name=str(normalized.at[idx, "metric_name"]),
            value=float(severity_num.at[idx]),
            message="Severity is non-integer.",
        )

    is_negative = severity_num.notna() & (severity_num < 0)
    for idx in normalized.index[is_negative]:
        _append_alert(
            alerts,
            "invalid_severity",
            row_index=int(idx),
            dataset=str(normalized.at[idx, "dataset"]),
            backbone=str(normalized.at[idx, "backbone"]),
            uq_method=str(normalized.at[idx, "uq_method"]),
            corruption_type=str(normalized.at[idx, "corruption_type"]),
            metric_name=str(normalized.at[idx, "metric_name"]),
            value=float(severity_num.at[idx]),
            message="Severity is negative.",
        )

    severity_valid = severity_num.notna() & ~is_non_integer & ~is_negative
    severity_int = pd.Series(pd.NA, index=normalized.index, dtype="Int64")
    severity_int.loc[severity_valid] = np.round(severity_num.loc[severity_valid]).astype(int)

    clean_mask = normalized["corruption_type"] == "clean"
    clean_bad = clean_mask & severity_int.notna() & (severity_int != 0)
    for idx in normalized.index[clean_bad]:
        _append_alert(
            alerts,
            "clean_severity_overridden",
            row_index=int(idx),
            dataset=str(normalized.at[idx, "dataset"]),
            backbone=str(normalized.at[idx, "backbone"]),
            uq_method=str(normalized.at[idx, "uq_method"]),
            corruption_type="clean",
            metric_name=str(normalized.at[idx, "metric_name"]),
            value=int(severity_int.at[idx]),
            message="Clean rows must use severity 0; value was overridden.",
        )
    severity_int.loc[clean_mask] = 0
    normalized["severity_int"] = severity_int

    metric_value_num = pd.to_numeric(normalized["metric_value"], errors="coerce")
    normalized["metric_value_num"] = metric_value_num
    non_finite_mask = ~np.isfinite(metric_value_num.to_numpy(dtype=float))
    for idx in normalized.index[non_finite_mask]:
        _append_alert(
            alerts,
            "non_finite_metric_value",
            row_index=int(idx),
            dataset=str(normalized.at[idx, "dataset"]),
            backbone=str(normalized.at[idx, "backbone"]),
            uq_method=str(normalized.at[idx, "uq_method"]),
            corruption_type=str(normalized.at[idx, "corruption_type"]),
            severity=int(normalized.at[idx, "severity_int"])
            if pd.notna(normalized.at[idx, "severity_int"])
            else None,
            metric_name=str(normalized.at[idx, "metric_name"]),
            value=str(normalized.at[idx, "metric_value"]),
            message="Metric value is NaN or non-finite.",
        )

    _run_range_checks(normalized, alerts)
    _run_duplicate_check(normalized, alerts)
    _run_missing_metric_check(normalized, alerts)

    return normalized, alerts


def _run_range_checks(df: pd.DataFrame, alerts: list[dict[str, object]]) -> None:
    finite = df["metric_value_num"].replace([np.inf, -np.inf], np.nan).notna()
    if not finite.any():
        return

    checks: dict[str, tuple[float | None, float | None]] = {
        "accuracy": (0.0, 1.0),
        "ece": (0.0, None),
        "brier": (0.0, None),
        "empirical_coverage": (0.0, 1.0),
        "mean_set_size": (0.0, None),
    }
    for metric, (min_value, max_value) in checks.items():
        mask = finite & (df["metric_name"] == metric)
        if min_value is not None:
            mask &= df["metric_value_num"] < min_value
        if max_value is not None:
            mask |= (finite & (df["metric_name"] == metric) & (df["metric_value_num"] > max_value))
        for idx in df.index[mask]:
            _append_alert(
                alerts,
                "invalid_metric_range",
                row_index=int(idx),
                dataset=str(df.at[idx, "dataset"]),
                backbone=str(df.at[idx, "backbone"]),
                uq_method=str(df.at[idx, "uq_method"]),
                corruption_type=str(df.at[idx, "corruption_type"]),
                severity=int(df.at[idx, "severity_int"]) if pd.notna(df.at[idx, "severity_int"]) else None,
                metric_name=metric,
                value=float(df.at[idx, "metric_value_num"]),
                message=f"Metric '{metric}' outside expected range.",
            )


def _run_duplicate_check(df: pd.DataFrame, alerts: list[dict[str, object]]) -> None:
    key_cols = ["dataset", "backbone", "uq_method", "corruption_type", "severity_int", "metric_name"]
    grouped = df.groupby(key_cols, dropna=False, as_index=False).size()
    duplicates = grouped[grouped["size"] > 1]
    for _, row in duplicates.iterrows():
        _append_alert(
            alerts,
            "duplicate_rows",
            dataset=str(row["dataset"]),
            backbone=str(row["backbone"]),
            uq_method=str(row["uq_method"]),
            corruption_type=str(row["corruption_type"]),
            severity=int(row["severity_int"]) if pd.notna(row["severity_int"]) else None,
            metric_name=str(row["metric_name"]),
            value=int(row["size"]),
            message="Duplicate rows for key (dataset, backbone, uq_method, corruption_type, severity, metric).",
        )


def _run_missing_metric_check(df: pd.DataFrame, alerts: list[dict[str, object]]) -> None:
    key_cols = ["dataset", "backbone", "uq_method", "corruption_type", "severity_int"]
    valid = df[df["severity_int"].notna()]
    for key_vals, group in valid.groupby(key_cols, dropna=False):
        dataset, backbone, uq_method, corruption_type, severity = key_vals
        method = str(uq_method)
        if method in PROBABILISTIC_METHODS:
            expected = set(PROBABILISTIC_ALL_METRICS)
        elif method == CONFORMAL_METHOD:
            expected = set(CONFORMAL_METRICS)
        else:
            continue

        present = set(group["metric_name"].dropna().astype(str).tolist())
        missing = sorted(expected - present)
        if not missing:
            continue
        _append_alert(
            alerts,
            "missing_expected_metrics",
            dataset=str(dataset),
            backbone=str(backbone),
            uq_method=method,
            corruption_type=str(corruption_type),
            severity=int(severity) if pd.notna(severity) else None,
            value=",".join(missing),
            message=f"Missing expected metrics for method family: {', '.join(missing)}.",
        )


def _write_diagnostics(
    alerts: list[dict[str, object]],
    summary_rows: list[tuple[str, object]],
    diagnostics_dir: Path,
) -> tuple[Path, Path]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    alerts_path = diagnostics_dir / "alerts.csv"
    summary_path = diagnostics_dir / "summary.csv"

    alerts_df = pd.DataFrame(alerts, columns=DIAGNOSTIC_COLUMNS)
    alerts_df.to_csv(alerts_path, index=False)

    summary_df = pd.DataFrame(summary_rows, columns=["key", "value"])
    if not alerts_df.empty:
        per_type = alerts_df["alert_type"].value_counts().sort_index()
        counts = pd.DataFrame({"key": [f"alerts.{idx}" for idx in per_type.index], "value": per_type.values})
        summary_df = pd.concat([summary_df, counts], ignore_index=True)
    summary_df.to_csv(summary_path, index=False)
    return alerts_path, summary_path


def _filter_dataframe(
    df: pd.DataFrame,
    *,
    backbones: list[str] | None,
    datasets: list[str] | None,
    corruptions: list[str] | None,
) -> pd.DataFrame:
    filtered = df.copy()
    if backbones:
        filtered = filtered[filtered["backbone"].isin(backbones)]
    if datasets:
        filtered = filtered[filtered["dataset"].isin(datasets)]
    if corruptions:
        allowed = set(corruptions) | {"clean"}
        filtered = filtered[filtered["corruption_type"].isin(allowed)]
    return filtered


def _resolve_axis_values(
    values: Sequence[str],
    preferred_order: list[str] | None,
) -> list[str]:
    if preferred_order:
        present = set(values)
        return [item for item in preferred_order if item in present]
    return sorted(set(values))


def _resolve_present_order(values: Sequence[str], preferred_order: Sequence[str]) -> list[str]:
    present = set(values)
    return [item for item in preferred_order if item in present]


def _facet_figure(plt, nrows: int, ncols: int, *, width: float, height: float):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(max(width * ncols, 4), max(height * nrows, 3)))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])
    return fig, axes


def _plot_probabilistic_baselines(
    plt,
    df: pd.DataFrame,
    *,
    datasets: list[str],
    backbones: list[str],
    metrics: list[str],
    outdir: Path,
    fmt: str,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    generated: list[Path] = []
    skipped: list[str] = []
    baseline_df = df[
        (df["uq_method"] == "uncalibrated")
        & (df["corruption_type"] == "clean")
        & (df["severity_int"] == 0)
        & (df["metric_value_num"].replace([np.inf, -np.inf], np.nan).notna())
    ]
    outdir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        metric_df = baseline_df[baseline_df["metric_name"] == metric]
        if metric_df.empty:
            skipped.append(f"probabilistic/baseline {metric}: no data")
            continue
        pivot = metric_df.pivot_table(
            index="dataset",
            columns="backbone",
            values="metric_value_num",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=datasets, columns=backbones)
        values = pivot.to_numpy(dtype=float)
        if np.isnan(values).all():
            skipped.append(f"probabilistic/baseline {metric}: all values missing")
            continue

        fig, ax = plt.subplots(
            figsize=(max(1.6 * len(backbones), 5.5), max(1.1 * len(datasets), 2.8)),
        )
        masked = np.ma.masked_invalid(values)
        im = ax.imshow(masked, cmap="YlGnBu", aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.set_label(metric)

        ax.set_xticks(np.arange(len(backbones)))
        ax.set_xticklabels(backbones, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(datasets)))
        ax.set_yticklabels(datasets)
        ax.set_xlabel("backbone")
        ax.set_ylabel("dataset")
        ax.set_title(f"Uncalibrated clean baseline: {metric}", fontsize=12)

        finite_values = values[np.isfinite(values)]
        midpoint = float(np.median(finite_values)) if finite_values.size else 0.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                val = values[i, j]
                if np.isfinite(val):
                    text_color = "black" if val <= midpoint else "white"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=8)
                else:
                    ax.text(j, i, "—", ha="center", va="center", color="#666666", fontsize=10)

        fig.tight_layout()
        out_path = outdir / f"{_slugify(metric)}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated, skipped


def _plot_method_trends(
    plt,
    df: pd.DataFrame,
    *,
    datasets: list[str],
    backbones: list[str],
    metrics: list[str],
    corruption_types: list[str],
    methods: tuple[str, ...],
    colors: dict[str, str],
    markers: dict[str, str] | None,
    title_prefix: str,
    outdir: Path,
    fmt: str,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    generated: list[Path] = []
    skipped: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    finite_mask = df["metric_value_num"].replace([np.inf, -np.inf], np.nan).notna()
    base_df = df[finite_mask & df["severity_int"].notna()].copy()
    base_df["severity_int"] = base_df["severity_int"].astype(int)

    for metric in metrics:
        metric_df = base_df[base_df["metric_name"] == metric]
        if metric_df.empty:
            skipped.append(f"{title_prefix} {metric}: no rows")
            continue

        for corruption in corruption_types:
            if corruption == "clean":
                corr_df = metric_df[metric_df["corruption_type"] == "clean"]
            else:
                corr_df = metric_df[metric_df["corruption_type"].isin(["clean", corruption])]
            corr_df = corr_df[corr_df["uq_method"].isin(methods)]
            if corr_df.empty:
                skipped.append(f"{title_prefix} {metric} / {corruption}: no rows")
                continue

            fig, axes = _facet_figure(plt, len(datasets), len(backbones), width=4.0, height=3.0)
            has_any_line = False
            all_severities = sorted(corr_df["severity_int"].dropna().astype(int).unique().tolist())
            for i, dataset in enumerate(datasets):
                for j, backbone in enumerate(backbones):
                    ax = axes[i, j]
                    panel = corr_df[(corr_df["dataset"] == dataset) & (corr_df["backbone"] == backbone)]
                    if panel.empty:
                        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                    else:
                        for method in methods:
                            method_rows = panel[panel["uq_method"] == method]
                            if method_rows.empty:
                                continue
                            agg = (
                                method_rows.groupby("severity_int", as_index=False)["metric_value_num"]
                                .mean()
                                .sort_values("severity_int")
                            )
                            if agg.empty:
                                continue
                            ax.plot(
                                agg["severity_int"].to_numpy(dtype=int),
                                agg["metric_value_num"].to_numpy(dtype=float),
                                marker=markers.get(method, "o") if markers else "o",
                                markersize=3.5,
                                linewidth=1.6,
                                label=method,
                                color=colors[method],
                            )
                            has_any_line = True
                    ax.grid(alpha=0.25)
                    if all_severities:
                        ax.set_xticks(all_severities)
                    ax.set_xlabel("severity")
                    if i == 0:
                        ax.set_title(backbone, fontsize=10)
                    if j == 0:
                        ax.set_ylabel(f"{dataset}\n{metric}")
                    else:
                        ax.set_ylabel("")

            if not has_any_line:
                plt.close(fig)
                skipped.append(f"{title_prefix} {metric} / {corruption}: all panels empty")
                continue

            handles: list[object] = []
            labels: list[str] = []
            for ax in axes.ravel():
                panel_handles, panel_labels = ax.get_legend_handles_labels()
                if panel_handles:
                    handles = panel_handles
                    labels = panel_labels
                    break
            layout_top = 0.93
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.955),
                    ncol=min(len(handles), 4),
                    frameon=False,
                    fontsize=9,
                )
                layout_top = 0.87
            fig.suptitle(f"{title_prefix}: {metric} vs severity ({corruption})", fontsize=12, y=0.99)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
            out_path = outdir / f"{_slugify(metric)}__{_slugify(corruption)}.{fmt}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(out_path)

    return generated, skipped


def _prepare_dataset_trend_grid(
    df: pd.DataFrame,
    *,
    dataset: str,
    corruption_type: str,
    metrics: Sequence[str],
) -> tuple[pd.DataFrame, list[str], list[str], list[int]]:
    finite_mask = df["metric_value_num"].replace([np.inf, -np.inf], np.nan).notna()
    subset = df[
        finite_mask
        & df["severity_int"].notna()
        & (df["dataset"] == dataset)
        & (df["corruption_type"].isin(["clean", corruption_type]))
        & (df["metric_name"].isin(metrics))
    ].copy()
    if subset.empty:
        return subset, [], [], []

    subset["severity_int"] = subset["severity_int"].astype(int)
    methods = _resolve_present_order(subset["uq_method"].astype(str).tolist(), ALL_METHODS)
    models = sorted(subset["model_name"].dropna().astype(str).unique().tolist())
    severities = sorted(subset["severity_int"].dropna().astype(int).unique().tolist())
    return subset, methods, models, severities


def _plot_dataset_trend_grids(
    plt,
    df: pd.DataFrame,
    *,
    datasets: list[str],
    metrics: list[str],
    corruption_types: list[str],
    outdir: Path,
    fmt: str,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    generated: list[Path] = []
    skipped: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        for corruption in corruption_types:
            subset, methods, models, severities = _prepare_dataset_trend_grid(
                df,
                dataset=dataset,
                corruption_type=corruption,
                metrics=metrics,
            )
            if subset.empty:
                skipped.append(f"dataset-trends {dataset} / {corruption}: no rows")
                continue
            if not methods:
                skipped.append(f"dataset-trends {dataset} / {corruption}: no methods")
                continue
            if not models:
                skipped.append(f"dataset-trends {dataset} / {corruption}: no models")
                continue

            fig, axes = _facet_figure(plt, len(metrics), len(methods), width=3.8, height=2.9)
            model_colors = {model: plt.get_cmap("tab10")(idx % 10) for idx, model in enumerate(models)}
            has_any_line = False

            for i, metric in enumerate(metrics):
                for j, method in enumerate(methods):
                    ax = axes[i, j]
                    panel = subset[(subset["metric_name"] == metric) & (subset["uq_method"] == method)]
                    panel_has_line = False
                    for model in models:
                        model_rows = panel[panel["model_name"] == model]
                        if model_rows.empty:
                            continue
                        agg = (
                            model_rows.groupby("severity_int", as_index=False)["metric_value_num"]
                            .mean()
                            .sort_values("severity_int")
                        )
                        if agg.empty:
                            continue
                        ax.plot(
                            agg["severity_int"].to_numpy(dtype=int),
                            agg["metric_value_num"].to_numpy(dtype=float),
                            marker="o",
                            markersize=3.2,
                            linewidth=1.5,
                            label=model,
                            color=model_colors[model],
                        )
                        panel_has_line = True
                        has_any_line = True
                    if not panel_has_line:
                        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                    ax.grid(alpha=0.25)
                    if severities:
                        ax.set_xticks(severities)
                    if i == 0:
                        ax.set_title(method, fontsize=10)
                    if j == 0:
                        ax.set_ylabel(metric)
                    else:
                        ax.set_ylabel("")
                    if i == len(metrics) - 1:
                        ax.set_xlabel("severity")
                    else:
                        ax.set_xlabel("")

            if not has_any_line:
                plt.close(fig)
                skipped.append(f"dataset-trends {dataset} / {corruption}: all panels empty")
                continue

            handles: list[object] = []
            labels: list[str] = []
            for ax in axes.ravel():
                panel_handles, panel_labels = ax.get_legend_handles_labels()
                if panel_handles:
                    handles = panel_handles
                    labels = panel_labels
                    break
            layout_top = 0.93
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.965),
                    ncol=min(5, len(handles)),
                    frameon=False,
                    fontsize=8,
                )
                layout_top = 0.88
            fig.suptitle(
                f"Dataset trends: {dataset} ({corruption})",
                fontsize=12,
                y=0.995,
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
            out_path = outdir / f"{_slugify(dataset)}__{_slugify(corruption)}.{fmt}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            generated.append(out_path)

    return generated, skipped


def _plot_calibration_by_severity(
    plt,
    df: pd.DataFrame,
    *,
    datasets: list[str],
    backbones: list[str],
    corruption_types: list[str],
    methods: tuple[str, ...],
    metric: str,
    outdir: Path,
    fmt: str,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    """Plot per-severity calibration summaries with one line per backbone."""
    generated: list[Path] = []
    skipped: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    finite_mask = df["metric_value_num"].replace([np.inf, -np.inf], np.nan).notna()
    base_df = df[finite_mask & df["severity_int"].notna()].copy()
    base_df["severity_int"] = base_df["severity_int"].astype(int)
    base_df = base_df[(base_df["metric_name"] == metric) & (base_df["uq_method"].isin(methods))]
    if base_df.empty:
        skipped.append(f"probabilistic/calibration {metric}: no rows")
        return generated, skipped

    method_order = [method for method in methods if method in set(base_df["uq_method"].astype(str).tolist())]
    if not method_order:
        skipped.append(f"probabilistic/calibration {metric}: no probabilistic methods present")
        return generated, skipped
    method_positions = np.arange(len(method_order), dtype=int)

    backbone_colors = {
        backbone: plt.get_cmap("tab20")(idx % 20) for idx, backbone in enumerate(backbones)
    }

    conditions: list[tuple[str, int, pd.DataFrame]] = []
    clean_df = base_df[(base_df["corruption_type"] == "clean") & (base_df["severity_int"] == 0)]
    if not clean_df.empty:
        conditions.append(("clean", 0, clean_df))
    for corruption in corruption_types:
        corr_df = base_df[base_df["corruption_type"] == corruption]
        if corr_df.empty:
            continue
        for severity in sorted(corr_df["severity_int"].unique().tolist()):
            if int(severity) <= 0:
                continue
            sev_df = corr_df[corr_df["severity_int"] == int(severity)]
            if sev_df.empty:
                continue
            conditions.append((corruption, int(severity), sev_df))

    if not conditions:
        skipped.append(f"probabilistic/calibration {metric}: no clean/corruption severities present")
        return generated, skipped

    ncols = min(3, max(1, len(datasets)))
    nrows = int(np.ceil(len(datasets) / ncols))
    for corruption, severity, condition_df in conditions:
        fig, axes = _facet_figure(plt, nrows, ncols, width=4.2, height=3.2)
        flat_axes = axes.ravel()
        has_any_line = False

        for idx, dataset in enumerate(datasets):
            ax = flat_axes[idx]
            panel = condition_df[condition_df["dataset"] == dataset]
            if panel.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                ax.set_xticks(method_positions)
                ax.set_xticklabels(method_order, rotation=25, ha="right")
                ax.grid(alpha=0.25)
                ax.set_title(dataset, fontsize=10)
                if idx % ncols == 0:
                    ax.set_ylabel(metric)
                continue

            for backbone in backbones:
                backbone_rows = panel[panel["backbone"] == backbone]
                if backbone_rows.empty:
                    continue
                agg = backbone_rows.groupby("uq_method", as_index=False)["metric_value_num"].mean()
                agg_map = dict(zip(agg["uq_method"].astype(str).tolist(), agg["metric_value_num"].tolist()))
                values = np.array([agg_map.get(method, np.nan) for method in method_order], dtype=float)
                valid = np.isfinite(values)
                if not valid.any():
                    continue
                ax.plot(
                    method_positions[valid],
                    values[valid],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.6,
                    label=backbone,
                    color=backbone_colors[backbone],
                )
                has_any_line = True

            ax.set_xticks(method_positions)
            ax.set_xticklabels(method_order, rotation=25, ha="right")
            ax.grid(alpha=0.25)
            ax.set_title(dataset, fontsize=10)
            if idx % ncols == 0:
                ax.set_ylabel(metric)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("uq_method")

        for idx in range(len(datasets), len(flat_axes)):
            flat_axes[idx].set_visible(False)

        if not has_any_line:
            plt.close(fig)
            skipped.append(
                f"probabilistic/calibration {metric} / {corruption} severity={severity}: all panels empty"
            )
            continue

        handles: list[object] = []
        labels: list[str] = []
        for ax in flat_axes:
            panel_handles, panel_labels = ax.get_legend_handles_labels()
            if panel_handles:
                handles = panel_handles
                labels = panel_labels
                break
        layout_top = 0.93
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=min(5, len(handles)),
                frameon=False,
                fontsize=9,
            )
            layout_top = 0.87
        fig.suptitle(
            f"Calibration summary ({metric}): corruption={corruption}, severity={severity}",
            fontsize=12,
            y=0.99,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
        out_path = outdir / f"{_slugify(metric)}__{_slugify(corruption)}__severity_{severity}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated, skipped


def _plot_confidence_trends(
    plt,
    df: pd.DataFrame,
    *,
    datasets: list[str],
    backbones: list[str],
    corruption_types: list[str],
    methods: tuple[str, ...],
    colors: dict[str, str],
    markers: dict[str, str] | None,
    outdir: Path,
    fmt: str,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    """Overlay accuracy and max_probability per method to reveal overconfidence under corruption.

    A growing gap between the solid accuracy line and the dashed confidence line indicates
    the model is becoming overconfident as corruption severity increases.
    """
    generated: list[Path] = []
    skipped: list[str] = []
    outdir.mkdir(parents=True, exist_ok=True)

    finite_mask = df["metric_value_num"].replace([np.inf, -np.inf], np.nan).notna()
    base_df = df[finite_mask & df["severity_int"].notna()].copy()
    base_df["severity_int"] = base_df["severity_int"].astype(int)
    base_df = base_df[base_df["metric_name"].isin(["accuracy", "max_probability"])]
    base_df = base_df[base_df["uq_method"].isin(methods)]

    if base_df.empty:
        skipped.append("confidence trends: no accuracy or max_probability rows")
        return generated, skipped

    group_cols = ["dataset", "backbone", "uq_method", "corruption_type", "severity_int"]
    pivoted = (
        base_df.groupby(group_cols + ["metric_name"], as_index=False)["metric_value_num"]
        .mean()
        .pivot_table(index=group_cols, columns="metric_name", values="metric_value_num")
        .reset_index()
    )
    pivoted.columns.name = None

    for corruption in corruption_types:
        if corruption == "clean":
            corr_df = pivoted[pivoted["corruption_type"] == "clean"]
        else:
            corr_df = pivoted[pivoted["corruption_type"].isin(["clean", corruption])]
        if corr_df.empty or ("accuracy" not in corr_df.columns and "max_probability" not in corr_df.columns):
            skipped.append(f"confidence trends / {corruption}: no rows after pivot")
            continue

        fig, axes = _facet_figure(plt, len(datasets), len(backbones), width=4.0, height=3.0)
        has_any_line = False
        all_severities = sorted(corr_df["severity_int"].dropna().astype(int).unique().tolist())

        for i, dataset in enumerate(datasets):
            for j, backbone in enumerate(backbones):
                ax = axes[i, j]
                panel = corr_df[(corr_df["dataset"] == dataset) & (corr_df["backbone"] == backbone)]
                if panel.empty:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                else:
                    for method in methods:
                        method_rows = panel[panel["uq_method"] == method].sort_values("severity_int")
                        if method_rows.empty:
                            continue
                        sevs = method_rows["severity_int"].to_numpy(dtype=int)
                        color = colors[method]
                        marker = markers.get(method, "o") if markers else "o"

                        if "accuracy" in method_rows.columns:
                            acc = method_rows["accuracy"].to_numpy(dtype=float)
                            valid = np.isfinite(acc)
                            if valid.any():
                                ax.plot(
                                    sevs[valid], acc[valid],
                                    linestyle="-", marker=marker, markersize=3.5,
                                    linewidth=1.6, color=color, label=method,
                                )
                                has_any_line = True

                        if "max_probability" in method_rows.columns:
                            conf = method_rows["max_probability"].to_numpy(dtype=float)
                            valid_c = np.isfinite(conf)
                            if valid_c.any():
                                ax.plot(
                                    sevs[valid_c], conf[valid_c],
                                    linestyle="--", marker=marker, markersize=3.5,
                                    linewidth=1.4, color=color,
                                )
                                has_any_line = True

                        # shade the overconfidence gap between confidence and accuracy
                        if "accuracy" in method_rows.columns and "max_probability" in method_rows.columns:
                            acc = method_rows["accuracy"].to_numpy(dtype=float)
                            conf = method_rows["max_probability"].to_numpy(dtype=float)
                            valid_both = np.isfinite(acc) & np.isfinite(conf)
                            if valid_both.any():
                                ax.fill_between(
                                    sevs[valid_both], acc[valid_both], conf[valid_both],
                                    alpha=0.07, color=color,
                                )

                ax.set_ylim(0.0, 1.05)
                ax.grid(alpha=0.25)
                if all_severities:
                    ax.set_xticks(all_severities)
                ax.set_xlabel("severity")
                if i == 0:
                    ax.set_title(backbone, fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{dataset}\nacc / confidence")
                else:
                    ax.set_ylabel("")

        if not has_any_line:
            plt.close(fig)
            skipped.append(f"confidence trends / {corruption}: all panels empty")
            continue

        # Build legend: method colors + linestyle explanation
        handles: list[object] = []
        labels: list[str] = []
        for ax in axes.ravel():
            panel_handles, panel_labels = ax.get_legend_handles_labels()
            if panel_handles:
                handles = panel_handles
                labels = panel_labels
                break

        import matplotlib.lines as mlines
        handles = list(handles)
        labels = list(labels)
        handles.append(mlines.Line2D([], [], color="gray", linestyle="-", linewidth=1.4))
        labels.append("accuracy (solid)")
        handles.append(mlines.Line2D([], [], color="gray", linestyle="--", linewidth=1.4))
        labels.append("max confidence (dashed)")

        layout_top = 0.93
        fig.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(handles), 4),
            frameon=False,
            fontsize=8,
        )
        layout_top = 0.87
        fig.suptitle(f"Confidence vs accuracy ({corruption})", fontsize=12, y=0.99)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))
        out_path = outdir / f"{_slugify(corruption)}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated, skipped


def _load_reliability_cache(trace_dir: Path) -> pd.DataFrame:
    parquet_path = trace_dir / "reliability_cache.parquet"
    csv_path = trace_dir / "reliability_cache.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Reliability cache not found under {trace_dir} (expected reliability_cache.parquet or .csv)."
    )


def _build_reliability_from_traces(
    trace_dir: Path,
    bins: int,
    binning: str,
    *,
    block_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    trace_df = scan_traces(
        trace_dir,
        block_keys=block_keys,
        columns=[
            "trace_block_key",
            "dataset",
            "backbone",
            "uq_method",
            "corruption_type",
            "severity",
            "model",
            "confidence",
            "correct",
        ],
    )
    if trace_df.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    group_cols = [
        "trace_block_key",
        "dataset",
        "backbone",
        "uq_method",
        "corruption_type",
        "severity",
        "model",
    ]
    for key_vals, block_df in trace_df.groupby(group_cols, dropna=False):
        entry = dict(zip(group_cols, key_vals, strict=False))
        conf = pd.to_numeric(block_df["confidence"], errors="coerce")
        corr = pd.to_numeric(block_df["correct"], errors="coerce")
        valid = conf.notna() & corr.notna()
        if valid.sum() == 0:
            continue
        rel_df = build_reliability_frame(
            confidence=conf[valid].to_numpy(dtype=float),
            correct=corr[valid].to_numpy(dtype=float),
            bins=bins,
            binning=binning,
        )
        for col in [
            "trace_block_key",
            "dataset",
            "backbone",
            "uq_method",
            "corruption_type",
            "severity",
            "model",
        ]:
            rel_df[col] = entry.get(col)
        rows.append(rel_df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _resolve_trace_dataset_root(csv_df: pd.DataFrame, trace_dir: Path | None) -> Path | None:
    if trace_dir is not None:
        return trace_dir
    if "trace_dataset_root" not in csv_df.columns:
        return None

    candidates = (
        csv_df["trace_dataset_root"].fillna("").astype(str).str.strip().replace("", pd.NA).dropna().unique()
    )
    if len(candidates) == 0:
        return None
    return Path(str(candidates[0]))


def _plot_reliability_from_cache(
    plt,
    reliability_df: pd.DataFrame,
    *,
    outdir: Path,
    fmt: str,
    dpi: int,
) -> tuple[list[Path], list[str]]:
    generated: list[Path] = []
    skipped: list[str] = []
    if reliability_df.empty:
        skipped.append("reliability: no rows")
        return generated, skipped

    outdir.mkdir(parents=True, exist_ok=True)
    reliability_df = reliability_df.copy()
    reliability_df["severity"] = pd.to_numeric(reliability_df["severity"], errors="coerce")
    reliability_df = reliability_df[reliability_df["severity"].notna()]
    reliability_df["severity"] = reliability_df["severity"].astype(int)

    group_cols = ["corruption_type", "severity"]
    for (corruption, severity), cond_df in reliability_df.groupby(group_cols, dropna=False):
        datasets = sorted(cond_df["dataset"].dropna().astype(str).unique().tolist())
        if not datasets:
            skipped.append(f"reliability {corruption} severity={severity}: no dataset rows")
            continue
        ncols = min(3, len(datasets))
        nrows = int(np.ceil(len(datasets) / ncols))
        fig, axes = _facet_figure(plt, nrows, ncols, width=4.2, height=3.2)
        flat_axes = axes.ravel()
        has_any = False

        for idx, dataset in enumerate(datasets):
            ax = flat_axes[idx]
            panel = cond_df[cond_df["dataset"].astype(str) == dataset]
            if panel.empty:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                continue
            for (backbone, method), line_df in panel.groupby(["backbone", "uq_method"], dropna=False):
                line_sorted = line_df.sort_values("bin_id")
                ax.plot(
                    line_sorted["mean_conf"].to_numpy(dtype=float),
                    line_sorted["accuracy"].to_numpy(dtype=float),
                    marker="o",
                    markersize=3.0,
                    linewidth=1.4,
                    label=f"{backbone}:{method}",
                )
                has_any = True
            ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#555555", alpha=0.7)
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_xlabel("mean_conf")
            if idx % ncols == 0:
                ax.set_ylabel("accuracy")
            ax.set_title(str(dataset), fontsize=10)
            ax.grid(alpha=0.25)

        for idx in range(len(datasets), len(flat_axes)):
            flat_axes[idx].set_visible(False)

        if not has_any:
            plt.close(fig)
            skipped.append(f"reliability {corruption} severity={severity}: all panels empty")
            continue

        handles, labels = flat_axes[0].get_legend_handles_labels()
        top = 0.93
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.955),
                ncol=min(4, len(handles)),
                frameon=False,
                fontsize=8,
            )
            top = 0.87
        fig.suptitle(f"Reliability: corruption={corruption}, severity={severity}", fontsize=12, y=0.99)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, top))
        out_path = outdir / f"reliability__{_slugify(str(corruption))}__severity_{int(severity)}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        generated.append(out_path)

    return generated, skipped


def _run(args: argparse.Namespace) -> int:
    if args.format.lower() != "png":
        raise ValueError("--format currently supports only 'png' in v1.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be a positive integer.")

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    logger.info("Reading %s", args.csv_path)
    input_df = pd.read_csv(args.csv_path)
    normalized_df, alerts = _normalize_frame(input_df)

    backbones_filter = _parse_csv_list(args.backbones)
    datasets_filter = _parse_csv_list(args.datasets)
    corruptions_filter = _parse_csv_list(args.corruptions)
    metrics_filter = _parse_csv_list(args.metrics)

    filtered_df = _filter_dataframe(
        normalized_df,
        backbones=backbones_filter,
        datasets=datasets_filter,
        corruptions=corruptions_filter,
    )

    if filtered_df.empty:
        logger.warning("No rows remaining after applying dataset/backbone/corruption filters.")

    summary_rows = [
        ("input_rows", len(input_df)),
        ("filtered_rows", len(filtered_df)),
        ("n_datasets", int(filtered_df["dataset"].nunique(dropna=True))),
        ("n_backbones", int(filtered_df["backbone"].nunique(dropna=True))),
        ("n_methods", int(filtered_df["uq_method"].nunique(dropna=True))),
        ("n_metrics", int(filtered_df["metric_name"].nunique(dropna=True))),
    ]

    diagnostics_dir = args.outdir / "diagnostics"
    alerts_path, summary_path = _write_diagnostics(alerts, summary_rows, diagnostics_dir)
    logger.info("Wrote diagnostics: %s, %s", alerts_path, summary_path)

    plt = _import_plotting()
    datasets = _resolve_axis_values(filtered_df["dataset"].astype(str).tolist(), datasets_filter)
    backbones = _resolve_axis_values(filtered_df["backbone"].astype(str).tolist(), backbones_filter)
    if not datasets or not backbones:
        logger.warning("No datasets/backbones available for plotting after filters.")
        return 0

    requested_metrics = metrics_filter or list(PROBABILISTIC_CORE_METRICS + CONFORMAL_METRICS)
    prob_metrics = [metric for metric in requested_metrics if metric in set(PROBABILISTIC_ALL_METRICS)]
    conformal_metrics = [metric for metric in requested_metrics if metric in set(CONFORMAL_METRICS)]
    if metrics_filter is None:
        prob_metrics = list(PROBABILISTIC_CORE_METRICS) + ["max_probability", "normalized_predictive_entropy"]
        conformal_metrics = list(CONFORMAL_METRICS)

    available_corruptions = sorted(
        c for c in filtered_df["corruption_type"].dropna().astype(str).unique().tolist() if c != "clean"
    )
    if corruptions_filter:
        trend_corruptions = [c for c in corruptions_filter if c == "clean" or c in available_corruptions]
    else:
        trend_corruptions = available_corruptions

    if not trend_corruptions:
        logger.warning("No trend corruptions selected or present (excluding clean).")
    dataset_grid_corruptions = [corruption for corruption in trend_corruptions if corruption != "clean"]

    prob_df = filtered_df[filtered_df["uq_method"].isin(PROBABILISTIC_METHODS)]
    conformal_df = filtered_df[filtered_df["uq_method"] == CONFORMAL_METHOD]

    generated_files: list[Path] = []
    skipped_messages: list[str] = []

    baseline_dir = args.outdir / "probabilistic" / "baseline"
    if prob_metrics:
        baseline_generated, baseline_skipped = _plot_probabilistic_baselines(
            plt,
            prob_df,
            datasets=datasets,
            backbones=backbones,
            metrics=prob_metrics,
            outdir=baseline_dir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(baseline_generated)
        skipped_messages.extend(baseline_skipped)
    else:
        logger.info("No probabilistic metrics selected for baseline plots.")

    prob_trends_dir = args.outdir / "probabilistic" / "trends"
    if prob_metrics and trend_corruptions:
        prob_generated, prob_skipped = _plot_method_trends(
            plt,
            prob_df,
            datasets=datasets,
            backbones=backbones,
            metrics=prob_metrics,
            corruption_types=trend_corruptions,
            methods=PROBABILISTIC_METHODS,
            colors=PROB_METHOD_COLORS,
            markers=PROB_METHOD_MARKERS,
            title_prefix="Probabilistic trends",
            outdir=prob_trends_dir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(prob_generated)
        skipped_messages.extend(prob_skipped)
    else:
        logger.info("Skipping probabilistic trend plots (no metrics and/or no corruptions).")

    calibration_metric = str(args.calibration_metric or "").strip().lower()
    calibration_dir = args.outdir / "probabilistic" / "calibration_by_severity"
    if calibration_metric:
        calibration_generated, calibration_skipped = _plot_calibration_by_severity(
            plt,
            prob_df,
            datasets=datasets,
            backbones=backbones,
            corruption_types=trend_corruptions,
            methods=PROBABILISTIC_METHODS,
            metric=calibration_metric,
            outdir=calibration_dir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(calibration_generated)
        skipped_messages.extend(calibration_skipped)
    else:
        logger.info("Skipping probabilistic calibration summary plots (metric disabled).")

    confidence_trends_dir = args.outdir / "probabilistic" / "confidence_trends"
    if not args.no_confidence_trends and trend_corruptions:
        ct_generated, ct_skipped = _plot_confidence_trends(
            plt,
            prob_df,
            datasets=datasets,
            backbones=backbones,
            corruption_types=trend_corruptions,
            methods=PROBABILISTIC_METHODS,
            colors=PROB_METHOD_COLORS,
            markers=PROB_METHOD_MARKERS,
            outdir=confidence_trends_dir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(ct_generated)
        skipped_messages.extend(ct_skipped)
    else:
        logger.info("Skipping confidence trend plots (disabled or no corruptions).")

    conformal_trends_dir = args.outdir / "conformal" / "trends"
    if conformal_metrics and trend_corruptions:
        conf_generated, conf_skipped = _plot_method_trends(
            plt,
            conformal_df,
            datasets=datasets,
            backbones=backbones,
            metrics=conformal_metrics,
            corruption_types=trend_corruptions,
            methods=(CONFORMAL_METHOD,),
            colors={CONFORMAL_METHOD: CONFORMAL_COLOR},
            markers=None,
            title_prefix="Conformal trends",
            outdir=conformal_trends_dir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(conf_generated)
        skipped_messages.extend(conf_skipped)
    else:
        logger.info("Skipping conformal trend plots (no metrics and/or no corruptions).")

    dataset_trends_dir = args.outdir / "by_dataset_trends"
    dataset_grid_metrics = [
        metric for metric in requested_metrics if metric in set(filtered_df["metric_name"].astype(str).tolist())
    ]
    if dataset_grid_metrics and dataset_grid_corruptions:
        dataset_generated, dataset_skipped = _plot_dataset_trend_grids(
            plt,
            filtered_df,
            datasets=datasets,
            metrics=dataset_grid_metrics,
            corruption_types=dataset_grid_corruptions,
            outdir=dataset_trends_dir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(dataset_generated)
        skipped_messages.extend(dataset_skipped)
    else:
        logger.info("Skipping dataset trend grids (no metrics and/or no non-clean corruptions).")

    trace_root = _resolve_trace_dataset_root(filtered_df, args.trace_dir)
    if trace_root is not None:
        reliability_outdir = (
            args.reliability_outdir
            if args.reliability_outdir is not None
            else args.outdir / "reliability"
        )
        if args.reliability_use_cache:
            reliability_df = _load_reliability_cache(trace_root)
        else:
            block_keys = None
            if "trace_block_key" in filtered_df.columns:
                keys = (
                    filtered_df["trace_block_key"]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )
                block_keys = keys or None
            reliability_df = _build_reliability_from_traces(
                trace_root,
                bins=int(args.reliability_bins),
                binning=str(args.reliability_binning),
                block_keys=block_keys,
            )
        rel_generated, rel_skipped = _plot_reliability_from_cache(
            plt,
            reliability_df,
            outdir=reliability_outdir,
            fmt=args.format.lower(),
            dpi=args.dpi,
        )
        generated_files.extend(rel_generated)
        skipped_messages.extend(rel_skipped)

    logger.info("Generated %d figure files.", len(generated_files))
    for path in generated_files:
        logger.info("  wrote %s", path)
    for message in skipped_messages:
        logger.info("  skipped: %s", message)
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
