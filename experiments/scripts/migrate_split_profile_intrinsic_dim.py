#!/usr/bin/env python3
"""One-off migration: split ``profile``/``intrinsic_dim`` rows out of results/models/.

Profile (throughput/latency/params) and intrinsic-dim rows are one-time
model+hardware measurements, unlike ``knn5``/``linear``/``seg-*`` rows which
change on every metrics rerun. Historically both lived in the same
``results/models/<name>.csv`` file, so a routine metrics rerun touched
(and diffed) the file holding these expensive one-off measurements too.

This script splits every ``results/models/<name>.csv`` by its ``method``
column:

- ``profile`` rows -> ``results/profiles/<name>.csv``
- ``intrinsic_dim`` rows -> ``results/intrinsic_dim/<name>.csv``
- everything else stays in ``results/models/<name>.csv``

Row order is preserved (no resorting) and each output side file's rows are
appended after any rows already there, so this is safe to rerun.

Usage::

    python experiments/scripts/migrate_split_profile_intrinsic_dim.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "results" / "models"
PROFILE_DIR = ROOT / "results" / "profiles"
INTRINSIC_DIM_DIR = ROOT / "results" / "intrinsic_dim"

SIDE_METHODS = {"profile": PROFILE_DIR, "intrinsic_dim": INTRINSIC_DIM_DIR}


def _read_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _append_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    if path.exists():
        existing_fieldnames, existing_rows = _read_rows(path)
        if existing_fieldnames != fieldnames:
            raise ValueError(
                f"schema mismatch appending to {path}: "
                f"existing={existing_fieldnames} new={fieldnames}"
            )
        _write_rows(path, fieldnames, existing_rows + rows)
    else:
        _write_rows(path, fieldnames, rows)


def migrate_one(path: Path) -> tuple[int, dict[str, int]]:
    """Split one models/<name>.csv file. Returns (kept, {method: moved_count})."""
    fieldnames, rows = _read_rows(path)
    kept_rows: list[dict] = []
    side_rows: dict[str, list[dict]] = {m: [] for m in SIDE_METHODS}

    for row in rows:
        method = row.get("method", "")
        if method in SIDE_METHODS:
            side_rows[method].append(row)
        else:
            kept_rows.append(row)

    _write_rows(path, fieldnames, kept_rows)
    moved_counts = {}
    for method, out_dir in SIDE_METHODS.items():
        out_path = out_dir / path.name
        _append_rows(out_path, fieldnames, side_rows[method])
        moved_counts[method] = len(side_rows[method])

    return len(kept_rows), moved_counts


def main() -> None:
    csv_paths = sorted(MODELS_DIR.glob("*.csv"))
    total_original = 0
    total_kept = 0
    total_moved: dict[str, int] = dict.fromkeys(SIDE_METHODS, 0)

    for path in csv_paths:
        _, original_rows = _read_rows(path)
        total_original += len(original_rows)
        kept, moved_counts = migrate_one(path)
        total_kept += kept
        for method, count in moved_counts.items():
            total_moved[method] += count
        if any(moved_counts.values()):
            print(f"{path.name}: kept={kept}, moved={moved_counts}")

    total_moved_all = sum(total_moved.values())
    print(
        f"\nTotal rows: original={total_original}, "
        f"kept in models/={total_kept}, moved={total_moved} "
        f"(sum moved={total_moved_all})"
    )
    if total_kept + total_moved_all != total_original:
        raise SystemExit(
            f"Row count mismatch: {total_kept} + {total_moved_all} != {total_original}"
        )
    print("Row counts conserved.")


if __name__ == "__main__":
    main()
