"""Regenerate ``docs/_static/results-explorer.html`` from result snapshots.

Reads ``results/models/*.csv``, ``results/profiles/*.csv``, and
``results/intrinsic_dim/*.csv``, writes today's snapshot to
``docs/_static/_results_snapshots/<label>.json``, then re-inlines every
committed snapshot (newest first) into the explorer HTML and bumps the
masthead.  Keeps ``knn5`` / ``linear`` / ``profile`` rows; the explorer's
Compute & efficiency figure joins the latter against the former.

Usage::

    python experiments/scripts/regen_results_explorer.py [--label 2026-05-08]
"""

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from torchgeo_bench.results import load_results

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "models"
# Profile/intrinsic_dim rows live in their own per-model files, separate
# from the knn5/linear/seg metrics files under RESULTS_DIR.
PROFILE_RESULTS_DIR = ROOT / "results" / "profiles"
INTRINSIC_DIM_RESULTS_DIR = ROOT / "results" / "intrinsic_dim"
HTML_PATH = ROOT / "docs" / "_static" / "results-explorer.html"
SNAPSHOT_DIR = ROOT / "docs" / "_static" / "_results_snapshots"
ALLOWED_METHODS = ("knn5", "linear", "profile", "intrinsic_dim")

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
}
BOOL = {"merge_val"}


def _load_csv_rows(label: str) -> list[dict]:
    """Return snapshot rows from every per-model results CSV across all three dirs.

    Loading goes through :func:`load_results` so this page sees the same
    deduplicated, normalization-corrected rows as the leaderboard generator.
    """
    frames = [
        load_results(directory)
        for directory in (RESULTS_DIR, PROFILE_RESULTS_DIR, INTRINSIC_DIM_RESULTS_DIR)
    ]
    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True, sort=False)
    df = df[df["method"].isin(ALLOWED_METHODS) & df["metric_value"].notna()]

    data_columns = [column for column in COLUMNS if column != "snapshot"]
    for column in data_columns:
        if column not in df.columns:
            df[column] = None
    df = df[data_columns].copy()
    for column in NUMERIC:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in BOOL:
        df[column] = df[column].astype(str).str.lower().isin(("true", "1"))

    # JSON has no NaN literal the browser will parse, so every gap becomes null.
    rows = df.astype(object).where(df.notna(), None).to_dict("records")
    for row in rows:
        row["snapshot"] = label
    return rows


def _snapshot_label_sort_key(label: str) -> tuple:
    """Sort labels with leading ``YYYY-MM-DD`` chronologically; the rest lex."""
    m = re.match(r"(\d{4}-\d{2}-\d{2})", label)
    return (m.group(1) if m else "", label)


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

    accuracy_rows = [r for r in latest_rows if r["method"] in ("knn5", "linear")]
    n_models = len({r["name"] for r in accuracy_rows if r["name"]}) or len(
        {r["name"] for r in latest_rows if r["name"]}
    )
    n_datasets = len({r["dataset"] for r in latest_rows})
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

    # The masthead states what the snapshot contains.  It never names a
    # leader or reads a result: any such claim goes stale on the next sweep,
    # and interpreting the numbers is the reader's job, not the page's.
    text = re.sub(
        r'<h1 class="headline" id="headline-text">[^<]*</h1>',
        ('<h1 class="headline" id="headline-text">Torchgeo-Bench results explorer</h1>'),
        text,
    )
    text = re.sub(
        # The standfirst may carry <em> spans, so match to the closing tag.
        r'<p class="standfirst" id="standfirst-text">.*?</p>',
        (
            f'<p class="standfirst" id="standfirst-text">'
            f"Every measurement in this snapshot: {len(latest_rows):,} rows covering "
            f"{n_datasets} datasets and {n_models} frozen-backbone variants, each row "
            f"recording its probe, band set, normalization policy and bootstrapped 95% "
            f"confidence interval. Use the controls to filter; the figures and the "
            f"table follow the selection."
            "</p>"
        ),
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r'<b id="row-shown">\d+</b> of <b id="row-total">\d+</b>',
        f'<b id="row-shown">{len(latest_rows)}</b> of <b id="row-total">{len(latest_rows)}</b>',
        text,
    )
    text = re.sub(
        r"Source: <b>[^<]*</b>",
        "Source: <b>results/{models,profiles,intrinsic_dim}/*.csv</b>",
        text,
    )
    text = re.sub(
        r"Published <b>[^<]*</b>",
        f"Published <b>{date.today().strftime('%-d %B %Y')}</b>",
        text,
    )
    text = re.sub(
        r"documented in <code>[^<]*</code>\. Confidence intervals are 95%\s+bootstrap on test predictions \(default \d+ resamples\)\.",
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
