"""Regenerate ``docs/_static/results-explorer.html`` from result snapshots.

Reads ``results/all_results.csv``, writes today's snapshot to
``docs/_static/_results_snapshots/<label>.json``, then re-inlines every
committed snapshot (newest first) into the explorer HTML and bumps the
masthead.  Keeps ``knn5`` / ``linear`` / ``profile`` rows; the explorer's
Compute & efficiency figure joins the latter against the former.

The GPU price / carbon intensity tables under ``experiments/scripts/cost/`` are
inlined as JS constants so the explorer can extrapolate $ and kgCO2 per
1M inferences in-browser, without a separate fetch.

Usage::

    python experiments/scripts/regen_results_explorer.py [--label 2026-05-08]
"""

import argparse
import csv
import json
import re
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "all_results.csv"
HTML_PATH = ROOT / "docs" / "_static" / "results-explorer.html"
SNAPSHOT_DIR = ROOT / "docs" / "_static" / "_results_snapshots"
COST_DIR = ROOT / "scripts" / "cost"
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
    rows = []
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
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

    prices = yaml.safe_load((COST_DIR / "gpu_prices.yaml").read_text())["instances"]
    carbon = yaml.safe_load((COST_DIR / "carbon_intensity.yaml").read_text())["regions"]

    js_columns = "const COLUMNS = " + json.dumps(COLUMNS) + ";"
    js_numeric = "const NUMERIC_COLS = " + json.dumps(sorted(NUMERIC)) + ";"
    js_snaps = "const SNAPSHOTS = " + json.dumps(snapshot_meta) + ";"
    js_default = "const DEFAULT_SNAPSHOT = " + json.dumps(latest_label) + ";"
    js_data = "const DATA = " + json.dumps(flat_rows, separators=(",", ":")) + ";"
    js_prices = "const GPU_PRICES = " + json.dumps(prices, separators=(",", ":")) + ";"
    js_carbon = "const CARBON_INTENSITY = " + json.dumps(carbon, separators=(",", ":")) + ";"

    text = HTML_PATH.read_text()
    pattern = re.compile(
        r"const COLUMNS = \[.*?\];\s*const NUMERIC_COLS = \[.*?\];"
        r"(?:\s*const SNAPSHOTS = \[.*?\];)?(?:\s*const DEFAULT_SNAPSHOT = \"[^\"]*\";)?"
        r"\s*const DATA = \[.*?\];"
        r"(?:\s*const GPU_PRICES = \[.*?\];)?(?:\s*const CARBON_INTENSITY = \[.*?\];)?",
        re.DOTALL,
    )
    new_block = "\n".join(
        [js_columns, js_numeric, js_snaps, js_default, js_data, js_prices, js_carbon]
    )
    if not pattern.search(text):
        raise SystemExit("Could not locate COLUMNS/NUMERIC_COLS/DATA block in HTML.")
    text = pattern.sub(new_block, text, count=1)

    text = re.sub(
        r'<h1 class="headline" id="headline-text">[^<]*</h1>',
        (
            '<h1 class="headline" id="headline-text">'
            "Four winners on GeoBench: Panopticon on KNN, DINOv3-SAT and "
            "OlmoEarth on linear, Terramind on multispectral"
            "</h1>"
        ),
        text,
    )
    text = re.sub(
        r'<p class="standfirst" id="standfirst-text">[^<]*<em>[^<]*</em>[^<]*</p>',
        (
            f'<p class="standfirst" id="standfirst-text">'
            f"Across {len(latest_rows):,} measurements on {n_datasets} GeoBench "
            f"classification datasets and {n_models} frozen-backbone variants, four "
            f"distinct leaders emerge: <em>Panopticon</em> tops KNN-5 on most datasets, "
            f"<em>DINOv3-SAT</em> remains the strongest RGB-only linear probe, "
            f"<em>OlmoEarth</em> reaches 97.8% on <em>eurosat-spatial</em> and "
            f"97.6% on <em>m-eurosat</em>, and <em>Terramind</em> wins the "
            f"multispectral datasets when all MSI bands are available."
            "</p>"
        ),
        text,
    )
    text = re.sub(
        r'<b id="row-shown">\d+</b> of <b id="row-total">\d+</b>',
        f'<b id="row-shown">{len(latest_rows)}</b> of <b id="row-total">{len(latest_rows)}</b>',
        text,
    )
    text = re.sub(
        r"Source: <b>[^<]*</b>",
        "Source: <b>results/all_results.csv</b>",
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
