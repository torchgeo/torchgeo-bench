#!/usr/bin/env python
"""Plot error-detection precision-recall curves from persisted UQ traces."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from torchgeo_bench.uq.error_pr import compute_error_pr

logger = logging.getLogger(__name__)


def _import_plotting():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for plot_uq_error_curves. Install `torchgeo-bench[viz]`."
        ) from exc
    return plt


def _parse_csv_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True, help="Trace run directory (run_id=...).")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for plots and summary.")
    parser.add_argument(
        "--uncertainties",
        type=str,
        default="u_conf,u_ent,u_set",
        help="Comma-separated uncertainties from {u_conf,u_ent,u_set}.",
    )
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names.")
    parser.add_argument("--backbones", type=str, default=None, help="Comma-separated backbone names.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated UQ methods.")
    parser.add_argument("--corruptions", type=str, default=None, help="Comma-separated corruption names.")
    parser.add_argument("--format", type=str, default="png", help="Figure format (default: png).")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI.")
    return parser


def _load_manifest(trace_dir: Path) -> pd.DataFrame:
    manifest_path = trace_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return manifest


def _load_trace_frames(manifest: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in manifest.to_dict(orient="records"):
        trace_path = Path(str(row["trace_path"]))
        fmt = str(row.get("trace_format", "parquet")).lower().strip()
        if not trace_path.exists():
            logger.warning("Skipping missing trace file: %s", trace_path)
            continue
        if fmt == "parquet":
            frame = pd.read_parquet(trace_path)
        elif fmt == "csv":
            frame = pd.read_csv(trace_path)
        else:
            logger.warning("Skipping trace with unsupported format '%s': %s", fmt, trace_path)
            continue
        frames.append(frame)

    if not frames:
        raise ValueError("No trace files could be loaded.")
    return pd.concat(frames, ignore_index=True)


def _prepare_uncertainty_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "confidence" in out.columns:
        out["u_conf"] = 1.0 - pd.to_numeric(out["confidence"], errors="coerce")
    if "normalized_predictive_entropy" in out.columns:
        out["u_ent"] = pd.to_numeric(out["normalized_predictive_entropy"], errors="coerce")
    if "set_size" in out.columns:
        out["u_set"] = pd.to_numeric(out["set_size"], errors="coerce")
    return out


def _filter_df(
    df: pd.DataFrame,
    *,
    datasets: list[str] | None,
    backbones: list[str] | None,
    methods: list[str] | None,
    corruptions: list[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    if datasets:
        out = out[out["dataset"].astype(str).isin(datasets)]
    if backbones:
        out = out[out["backbone"].astype(str).isin(backbones)]
    if methods:
        out = out[out["uq_method"].astype(str).isin(methods)]
    if corruptions:
        out = out[out["corruption_type"].astype(str).isin(corruptions)]
    return out


def _plot_group_curves(
    plt,
    group_df: pd.DataFrame,
    *,
    group_name: str,
    uncertainties: list[str],
    out_path: Path,
    dpi: int,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(5.8, 4.3))

    target = pd.to_numeric(group_df["is_error"], errors="coerce")
    valid_target = target.notna()
    if valid_target.sum() < 2:
        plt.close(fig)
        return summary_rows

    target_np = target[valid_target].to_numpy(dtype=int)
    if np.unique(target_np).size < 2:
        plt.close(fig)
        return summary_rows

    any_curve = False
    for uncertainty in uncertainties:
        if uncertainty not in group_df.columns:
            continue
        scores = pd.to_numeric(group_df[uncertainty], errors="coerce")
        valid = valid_target & scores.notna()
        if valid.sum() < 2:
            continue
        y_true = target[valid].to_numpy(dtype=int)
        y_score = scores[valid].to_numpy(dtype=float)
        if np.unique(y_true).size < 2:
            continue

        metrics = compute_error_pr(is_error=y_true, uncertainty=y_score)
        precision = np.asarray(metrics["precision"], dtype=float)
        recall = np.asarray(metrics["recall"], dtype=float)
        ap = float(metrics["ap"])
        auroc = float(metrics["auroc"])

        ax.plot(recall, precision, linewidth=1.7, label=f"{uncertainty} (AP={ap:.3f})")
        any_curve = True
        summary_rows.append(
            {
                "group": group_name,
                "uncertainty": uncertainty,
                "ap": ap,
                "aupr": ap,
                "auroc": auroc,
                "n_samples": int(valid.sum()),
                "n_errors": int(y_true.sum()),
            }
        )

    if not any_curve:
        plt.close(fig)
        return summary_rows

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(group_name)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def _slugify(value: str) -> str:
    return "_".join(part for part in value.lower().replace("/", "_").replace(" ", "_").split("_") if part)


def _run(args: argparse.Namespace) -> int:
    if args.format.lower() != "png":
        raise ValueError("--format currently supports only 'png'.")

    manifest = _load_manifest(args.trace_dir)
    trace_df = _load_trace_frames(manifest)
    trace_df = _prepare_uncertainty_columns(trace_df)

    datasets = _parse_csv_list(args.datasets)
    backbones = _parse_csv_list(args.backbones)
    methods = _parse_csv_list(args.methods)
    corruptions = _parse_csv_list(args.corruptions)
    uncertainties = _parse_csv_list(args.uncertainties) or ["u_conf", "u_ent", "u_set"]

    filtered = _filter_df(
        trace_df,
        datasets=datasets,
        backbones=backbones,
        methods=methods,
        corruptions=corruptions,
    )
    if filtered.empty:
        logger.warning("No trace rows remain after filtering.")
        return 0

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    curves_dir = outdir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)

    plt = _import_plotting()

    group_cols = ["dataset", "backbone", "uq_method", "corruption_type", "severity"]
    summary: list[dict[str, object]] = []
    for key_vals, group in filtered.groupby(group_cols, dropna=False):
        dataset, backbone, method, corruption, severity = key_vals
        group_name = (
            f"dataset={dataset} backbone={backbone} method={method} "
            f"corruption={corruption} severity={severity}"
        )
        slug = _slugify(
            f"{dataset}__{backbone}__{method}__{corruption}__severity_{severity}"
        )
        out_path = curves_dir / f"{slug}.{args.format.lower()}"
        rows = _plot_group_curves(
            plt,
            group,
            group_name=group_name,
            uncertainties=uncertainties,
            out_path=out_path,
            dpi=args.dpi,
        )
        summary.extend(rows)

    summary_path = outdir / "error_curve_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    logger.info("Wrote %s (%d rows).", summary_path, len(summary))
    return 0


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
