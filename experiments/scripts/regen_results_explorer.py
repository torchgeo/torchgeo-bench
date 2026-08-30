"""Regenerate ``docs/_static/results-explorer.html`` from result snapshots.

Reads ``results/models/*.csv``, ``results/profiles/*.csv``,
``results/intrinsic_dim/*.csv`` and ``results/compute_cost.csv``, writes
today's snapshot to ``docs/_static/_results_snapshots/<label>.json``, then
re-inlines every committed snapshot (newest first) into the explorer HTML and
bumps the masthead.  Keeps classification (``knn5`` / ``linear``),
segmentation (``seg-*``), ``profile`` and ``intrinsic_dim`` rows; the
explorer's Compute & efficiency figure joins the profile rows against the
classification ones.

Usage::

    python experiments/scripts/regen_results_explorer.py [--label 2026-05-08]
"""

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "models"
# Profile/intrinsic_dim rows live in their own per-model files, separate
# from the knn5/linear/seg metrics files under RESULTS_DIR.
PROFILE_RESULTS_DIR = ROOT / "results" / "profiles"
INTRINSIC_DIM_RESULTS_DIR = ROOT / "results" / "intrinsic_dim"
# Cost/throughput moved out of results/profiles/ into one wide table; only
# imagestats was ever migrated, which is why the efficiency figure emptied out.
COMPUTE_COST_CSV = ROOT / "results" / "compute_cost.csv"
HTML_PATH = ROOT / "docs" / "_static" / "results-explorer.html"
SNAPSHOT_DIR = ROOT / "docs" / "_static" / "_results_snapshots"
SEG_METHODS = ("seg-linear", "seg-conv_block", "seg-fpn", "seg-dpt")
ACCURACY_METHODS = ("knn5", "linear")
ALLOWED_METHODS = (*ACCURACY_METHODS, *SEG_METHODS, "profile", "intrinsic_dim")

# compute_cost.csv is wide -- one row per measured config, one column per
# quantity.  The explorer wants the long ``method="profile"`` rows that
# results/profiles/*.csv used to supply, so melt these columns into rows.
COMPUTE_COST_METRICS = (
    "gflops_backbone",
    "gflops_head",
    "gflops_probe",
    "gflops_total",
    "params_backbone_m",
    "params_head_m",
    "params_probe_m",
    "throughput_samples_per_sec",
    "latency_ms_per_batch_p50",
    "peak_gpu_mem_gb",
    "reserved_gpu_mem_gb",
    "n_tokens",
)
# The results CSVs call the multispectral run "all"; compute_cost measures it as
# the 12-band S2 stack.  Pair them so the efficiency join lands -- n_channels
# rides along on the row so a dataset whose "all" is not 12 bands stays visible.
BAND_CONFIG_ALIASES = {"s2": "all"}

COLUMNS = [
    "dataset",
    "method",
    "metric_name",
    "metric_value",
    "ci_lower",
    "ci_upper",
    "feature_dim",
    "best_c",
    "best_lr",
    "best_batch_size",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "model",
    "name",
    "normalization",
    "image_size",
    "interpolation",
    "partition",
    "bands",
    "c_range_start",
    "c_range_stop",
    "c_range_num",
    "merge_val",
    "bootstrap",
    "fw_iou",
    "precision",
    "recall",
    "f1",
    "task",
    "head_type",
    "n_channels",
    "snapshot",
]
NUMERIC = {
    "metric_value",
    "ci_lower",
    "ci_upper",
    "feature_dim",
    "best_c",
    "best_lr",
    "best_batch_size",
    "n_train",
    "n_val",
    "n_test",
    "seed",
    "image_size",
    "c_range_start",
    "c_range_stop",
    "c_range_num",
    "bootstrap",
    "fw_iou",
    "precision",
    "recall",
    "f1",
    "n_channels",
}
BOOL = {"merge_val"}


def _iter_compute_cost_rows():
    """Melt ``results/compute_cost.csv`` into long ``method="profile"`` rows.

    Cost is measured per (model, band config, task, head) at a fixed 224px
    input -- it carries no ``dataset``, because a frozen backbone costs the
    same whichever dataset is probed.  The explorer joins these to accuracy on
    (name, bands) for that reason.
    """
    if not COMPUTE_COST_CSV.exists():
        return
    with COMPUTE_COST_CSV.open() as fh:
        for r in csv.DictReader(fh):
            bands = BAND_CONFIG_ALIASES.get(r["band_config"], r["band_config"])
            for metric in COMPUTE_COST_METRICS:
                if not r.get(metric):
                    continue
                yield {
                    "dataset": "",
                    "method": "profile",
                    "metric_name": metric,
                    "metric_value": r[metric],
                    "model": r["model"],
                    "name": r["name"],
                    "bands": bands,
                    "n_channels": r["n_channels"],
                    "image_size": r["image_size"],
                    "feature_dim": r["feature_dim"],
                    "task": r["task"],
                    "head_type": r["head_type"],
                }


def _iter_result_rows():
    """Yield rows from every per-model results CSV, plus melted compute costs."""
    paths = sorted(RESULTS_DIR.glob("*.csv"))
    paths += sorted(PROFILE_RESULTS_DIR.glob("*.csv"))
    paths += sorted(INTRINSIC_DIM_RESULTS_DIR.glob("*.csv"))
    for path in paths:
        with path.open() as fh:
            yield from csv.DictReader(fh)
    yield from _iter_compute_cost_rows()


def _load_csv_rows(label: str) -> list[dict]:
    rows = []
    for r in _iter_result_rows():
        if r["method"] not in ALLOWED_METHODS:
            continue
        if not r.get("metric_value"):
            continue
        row = {}
        for k in COLUMNS:
            if k == "snapshot":
                continue
            v = r.get(k, "")
            if v is None or v == "":
                row[k] = None
            elif k in NUMERIC:
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = None
            elif k in BOOL:
                row[k] = v.lower() in ("true", "1")
            else:
                row[k] = v
        row["snapshot"] = label
        rows.append(row)
    return rows


def _snapshot_label_sort_key(label: str) -> tuple:
    """Sort labels with leading ``YYYY-MM-DD`` chronologically; the rest lex."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", label)
    return (m.group(1) if m else "", label)


def _mean_rank_leader(rows: list[dict], method: str) -> str | None:
    """Model with the best mean rank for ``method``, over models covering every dataset.

    Ranking (rather than mean metric) keeps datasets on different metric scales
    comparable; restricting to full coverage stops a model that ran on one easy
    dataset from outranking one evaluated everywhere.
    """
    best: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["method"] != method or r["metric_value"] is None or not r["name"]:
            continue
        scores = best.setdefault(r["dataset"], {})
        scores[r["name"]] = max(r["metric_value"], scores.get(r["name"], r["metric_value"]))
    if not best:
        return None
    ranks: dict[str, list[int]] = {}
    for scores in best.values():
        for rank, (name, _) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]), 1):
            ranks.setdefault(name, []).append(rank)
    full = {n: sum(v) / len(v) for n, v in ranks.items() if len(v) == len(best)}
    return min(full, key=lambda n: full[n]) if full else None


def _sub_once(pattern: str, repl: str, text: str) -> str:
    """Substitute exactly once, failing loudly if the anchor has drifted.

    A silent no-match here leaves stale copy on a page whose data has been
    refreshed -- the failure mode that left the masthead quoting May's numbers.
    """
    out, n = re.subn(pattern, lambda _: repl, text, count=1, flags=re.DOTALL)
    if n != 1:
        raise SystemExit(f"Could not locate {pattern!r} in {HTML_PATH.name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default=date.today().isoformat(),
        help="Label for the snapshot generated from the current CSV (default: today).",
    )
    args = parser.parse_args()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    today_rows = _load_csv_rows(args.label)
    snapshot_path = SNAPSHOT_DIR / f"{args.label}.json"
    snapshot_path.write_text(json.dumps(today_rows, separators=(",", ":")))

    snapshots: dict[str, list[dict]] = {}
    for path in sorted(SNAPSHOT_DIR.glob("*.json")):
        label = path.stem
        rows = json.loads(path.read_text())
        for r in rows:
            r["snapshot"] = label  # normalise even if file omits it
        snapshots[label] = rows

    ordered_labels = sorted(snapshots, key=_snapshot_label_sort_key, reverse=True)
    latest_label = ordered_labels[0]
    latest_rows = snapshots[latest_label]
    flat_rows = [r for label in ordered_labels for r in snapshots[label]]
    snapshot_meta = [{"label": label, "rows": len(snapshots[label])} for label in ordered_labels]

    accuracy_rows = [r for r in latest_rows if r["method"] in ACCURACY_METHODS]
    seg_rows = [r for r in latest_rows if r["method"] in SEG_METHODS]
    n_models = len({r["name"] for r in accuracy_rows if r["name"]}) or len(
        {r["name"] for r in latest_rows if r["name"]}
    )
    # Profile rows carry no dataset, so count datasets per task rather than
    # over every row.
    n_datasets = len({r["dataset"] for r in accuracy_rows if r["dataset"]})
    n_seg_datasets = len({r["dataset"] for r in seg_rows if r["dataset"]})
    n_seg_models = len({r["name"] for r in seg_rows if r["name"]})
    best = max(accuracy_rows or latest_rows, key=lambda r: r["metric_value"] or 0)

    js_columns = "const COLUMNS = " + json.dumps(COLUMNS) + ";"
    js_numeric = "const NUMERIC_COLS = " + json.dumps(sorted(NUMERIC)) + ";"
    js_snaps = "const SNAPSHOTS = " + json.dumps(snapshot_meta) + ";"
    js_default = "const DEFAULT_SNAPSHOT = " + json.dumps(latest_label) + ";"
    js_data = "const DATA = " + json.dumps(flat_rows, separators=(",", ":")) + ";"

    text = HTML_PATH.read_text()
    pattern = re.compile(
        r"const COLUMNS = \[.*?\];\s*const NUMERIC_COLS = \[.*?\];"
        r"(?:\s*const SNAPSHOTS = \[.*?\];)?(?:\s*const DEFAULT_SNAPSHOT = \"[^\"]*\";)?"
        r"\s*const DATA = \[.*?\];"
        r"(?:\s*const GPU_PRICES = \[.*?\];)?(?:\s*const CARBON_INTENSITY = \[.*?\];)?",
        re.DOTALL,
    )
    new_block = "\n".join([js_columns, js_numeric, js_snaps, js_default, js_data])
    if not pattern.search(text):
        raise SystemExit("Could not locate COLUMNS/NUMERIC_COLS/DATA block in HTML.")
    text = pattern.sub(new_block, text, count=1)

    knn_leader = _mean_rank_leader(accuracy_rows, "knn5")
    linear_leader = _mean_rank_leader(accuracy_rows, "linear")
    if knn_leader and linear_leader:
        if knn_leader == linear_leader:
            lede = f"{knn_leader} leads both KNN-5 and linear probing"
        else:
            lede = f"{knn_leader} leads KNN-5, {linear_leader} leads linear probing"
        detail = (
            f"<em>{knn_leader}</em> has the best mean rank under KNN-5 and "
            f"<em>{linear_leader}</em> under linear probing"
        )
    else:
        lede = f"{n_models} frozen backbones on {n_datasets} datasets"
        detail = "no model is evaluated on every dataset, so no overall rank is reported"

    seg_sentence = (
        f" A further {len(seg_rows):,} segmentation measurements cover "
        f"{n_seg_models} backbones on {n_seg_datasets} datasets (mIoU, four "
        f"decoder heads)."
        if seg_rows
        else ""
    )

    text = _sub_once(
        r'<h1 class="headline" id="headline-text">.*?</h1>',
        (
            f'<h1 class="headline" id="headline-text">'
            f"{lede} across {n_datasets} classification datasets"
            f"</h1>"
        ),
        text,
    )
    text = _sub_once(
        r'<p class="standfirst" id="standfirst-text">.*?</p>',
        (
            f'<p class="standfirst" id="standfirst-text">'
            f"Across {len(latest_rows):,} measurements on {n_datasets} "
            f"classification datasets and {n_models} frozen-backbone variants, "
            f"{detail}. The highest single score is "
            f"{best['metric_value']:.3f} ({best['metric_name']}) for "
            f"<em>{best['name']}</em> on <em>{best['dataset']}</em>."
            f"{seg_sentence}"
            f"</p>"
        ),
        text,
    )
    text = _sub_once(
        r'<b id="row-shown">\d+</b> of <b id="row-total">\d+</b>',
        f'<b id="row-shown">{len(latest_rows)}</b> of <b id="row-total">{len(latest_rows)}</b>',
        text,
    )
    text = _sub_once(
        r"Source: <b>[^<]*</b>",
        "Source: <b>results/{models,profiles,intrinsic_dim}/*.csv, results/compute_cost.csv</b>",
        text,
    )
    text = _sub_once(
        r"Published <b>[^<]*</b>",
        f"Published <b>{date.today().strftime('%-d %B %Y')}</b>",
        text,
    )
    text = _sub_once(
        r"documented in <code>[^<]*</code>\.\s+Confidence intervals\s+are 95%\s+bootstrap"
        r"\s+on test predictions\s+\(default \d+ resamples\)\.",
        (
            "documented in <code>docs/user/methodology.rst</code>. "
            "Confidence intervals are 95% bootstrap on test predictions "
            "(default 200 resamples)."
        ),
        text,
    )

    HTML_PATH.write_text(text)
    print(
        f"Wrote {HTML_PATH.relative_to(ROOT)}: "
        f"{len(snapshot_meta)} snapshots, latest={latest_label} "
        f"({len(latest_rows)} rows, {n_models} models, {n_datasets} datasets) — "
        f"best {best['metric_value']:.4f} ({best['name']} on {best['dataset']})"
    )


if __name__ == "__main__":
    main()
