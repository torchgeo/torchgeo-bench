"""Regenerate ``docs/_static/results-explorer.html`` from ``results/all_results.csv``.

The explorer ships as a self-contained HTML file with the dataset inlined as a
``const DATA = [...]`` JS array.  Whenever ``all_results.csv`` changes the
inlined snapshot drifts and the deployed leaderboard becomes stale.  Run this
script to splice the current CSV into the page and refresh the masthead
counters.

Usage::

    python scripts/regen_results_explorer.py

Only ``method in {knn5, linear}`` rows are inlined (intrinsic-dimension rows
have no ``metric_value`` and clutter the appendix table).
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "all_results.csv"
HTML_PATH = ROOT / "docs" / "_static" / "results-explorer.html"

COLUMNS = [
    "dataset", "method", "metric_name", "metric_value", "ci_lower", "ci_upper",
    "feature_dim", "best_c", "best_lr", "best_batch_size", "n_train", "n_val",
    "n_test", "seed", "model", "name", "normalization", "image_size",
    "interpolation", "partition", "bands", "c_range_start", "c_range_stop",
    "c_range_num", "merge_val", "bootstrap", "scope",
]

# Wrapper classes added in PR #32 (Prithvi / TerraMind / Clay / CROMA /
# Panopticon).  Everything else (TIMM, RCF, ImageStats, DOFA, EarthLoc,
# ScaleMAE, Swin, ResNet, OlmoEarth, SAM3) was on main before the PR.
PR32_CLASSES = {
    "torchgeo_bench.models.TerraTorchClayBench",
    "torchgeo_bench.models.TerraTorchPrithviBench",
    "torchgeo_bench.models.TerraTorchTerraMindBench",
    "torchgeo_bench.models.TorchGeoCromaBench",
    "torchgeo_bench.models.TorchGeoPanopticonBench",
}
# Pre-existing wrapper class but new checkpoint config introduced by PR #32.
PR32_NAMES = {
    "tgeo_resnet50_s2all_moco",
}
NUMERIC = {
    "metric_value", "ci_lower", "ci_upper", "feature_dim", "best_c", "best_lr",
    "best_batch_size", "n_train", "n_val", "n_test", "seed", "image_size",
    "c_range_start", "c_range_stop", "c_range_num", "bootstrap", "fw_iou",
    "precision", "recall", "f1",
}
BOOL = {"merge_val"}


def _load_rows() -> list[dict]:
    rows = []
    with CSV_PATH.open() as fh:
        for r in csv.DictReader(fh):
            if r["method"] not in ("knn5", "linear"):
                continue
            if not r.get("metric_value"):
                continue
            row = {}
            for k in COLUMNS:
                if k == "scope":
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
            row["scope"] = (
                "pr_32"
                if row["model"] in PR32_CLASSES or row["name"] in PR32_NAMES
                else "pre_pr"
            )
            rows.append(row)
    return rows


def main() -> None:
    rows = _load_rows()
    n_models = len({r["name"] for r in rows if r["name"]})
    n_datasets = len({r["dataset"] for r in rows})
    best = max(rows, key=lambda r: r["metric_value"])

    js_columns = "const COLUMNS = " + json.dumps(COLUMNS) + ";"
    js_numeric = "const NUMERIC_COLS = " + json.dumps(sorted(NUMERIC)) + ";"
    js_data = "const DATA = " + json.dumps(rows, separators=(",", ":")) + ";"

    text = HTML_PATH.read_text()
    pattern = re.compile(
        r"const COLUMNS = \[.*?\];\s*const NUMERIC_COLS = \[.*?\];\s*const DATA = \[.*?\];",
        re.DOTALL,
    )
    new_block = "\n".join([js_columns, js_numeric, js_data])
    if not pattern.search(text):
        raise SystemExit("Could not locate COLUMNS/NUMERIC_COLS/DATA block in HTML.")
    text = pattern.sub(new_block, text, count=1)

    text = re.sub(
        r'<h1 class="headline" id="headline-text">[^<]*</h1>',
        f'<h1 class="headline" id="headline-text">How {n_models} frozen backbones perform on GeoBench V1</h1>',
        text,
    )
    text = re.sub(
        r'<p class="standfirst" id="standfirst-text">[^<]*<em>[^<]*</em>[^<]*</p>',
        (
            f'<p class="standfirst" id="standfirst-text">A snapshot of {len(rows):,} '
            f"measurements across {n_datasets} classification datasets (RGB and full "
            f"multispectral) and {n_models} model variants. The best configuration in "
            f"this run reaches {best['metric_value'] * 100:.1f}% on "
            f"<em>{best['dataset']}</em> — explore the data below.</p>"
        ),
        text,
    )
    text = re.sub(
        r'<b id="row-shown">\d+</b> of <b id="row-total">\d+</b>',
        f'<b id="row-shown">{len(rows)}</b> of <b id="row-total">{len(rows)}</b>',
        text,
    )
    text = re.sub(
        r"Source: <b>[^<]*</b>",
        "Source: <b>results/all_results.csv</b>",
        text,
    )

    HTML_PATH.write_text(text)
    print(
        f"Wrote {HTML_PATH.relative_to(ROOT)}: "
        f"{len(rows)} rows, {n_models} models, {n_datasets} datasets — "
        f"best {best['metric_value']:.4f} ({best['name']} on {best['dataset']})"
    )


if __name__ == "__main__":
    main()
