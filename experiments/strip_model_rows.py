"""Strip all rows for a given model from a sample-size sweep CSV.

Use when a sweep was run under the wrong model name/config and the stale rows
must be removed before re-running. ``append_rows_atomic`` (the writer used by
``torchgeo-bench sample-size``) is append-only and never deletes, so a re-run
under the corrected name leaves the old rows behind; this script removes them.

A timestamped ``.bak`` copy of the CSV is written before any change. The CSV is
matched on an exact value in the ``model`` column.

Run (after `source sc_venv_template/activate.sh`):
  python experiments/strip_model_rows.py --model tt_terramind_v1_base
  python experiments/strip_model_rows.py --model tt_terramind_v1_base \
      --csv results/sample_size_full.csv --dry-run
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_DEFAULT = Path("results/sample_size_full.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Exact value in the 'model' column to drop.")
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    ap.add_argument(
        "--dry-run", action="store_true", help="Report what would be dropped without writing."
    )
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "model" not in df.columns:
        raise SystemExit(f"No 'model' column in {args.csv} (columns: {list(df.columns)})")

    mask = df["model"] == args.model
    n_drop = int(mask.sum())
    if n_drop == 0:
        present = sorted(df["model"].unique())
        raise SystemExit(
            f"No rows with model == {args.model!r} in {args.csv}.\nPresent models: {present}"
        )

    print(f"{args.csv}: {len(df)} rows total, {n_drop} match model == {args.model!r}.")
    if args.dry_run:
        print("Dry run — nothing written.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = args.csv.with_suffix(args.csv.suffix + f".bak_{stamp}")
    shutil.copy2(args.csv, backup)
    print(f"Backup written to {backup}")

    df[~mask].to_csv(args.csv, index=False)
    print(f"Wrote {len(df) - n_drop} rows to {args.csv} (dropped {n_drop}).")


if __name__ == "__main__":
    main()
