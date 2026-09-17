#!/usr/bin/env python3
"""Generate AID's deterministic 60/20/20 train/val/test split.

No official AID split exists (Xia et al. 2017 does not publish one). This
stratifies each class's sorted filenames with a fixed seed and writes three
newline-delimited basename lists, mirroring the ``resisc45-{split}.txt``
convention torchgeo's RESISC45 downloader uses.

Usage::

    $ python scripts/generate_aid_splits.py --root data/aid/AID --output-dir /tmp/aid-splits
"""

import argparse
import logging
import random
import sys
from pathlib import Path

from torchvision.datasets.folder import IMG_EXTENSIONS

logger = logging.getLogger(__name__)

_TRAIN_FRACTION = 0.6
_VAL_FRACTION = 0.2
_SPLIT_SEED = 0


def split_for_class(filenames: list[str]) -> dict[str, set[str]]:
    """Stratify one class's sorted filenames into train/val/test by ``_SPLIT_SEED``."""
    ordered = sorted(filenames)
    rng = random.Random(_SPLIT_SEED)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = round(n * _TRAIN_FRACTION)
    n_val = round(n * _VAL_FRACTION)
    return {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train : n_train + n_val]),
        "test": set(ordered[n_train + n_val :]),
    }


def basenames_for_split(root: Path, split: str) -> set[str]:
    """Return every image basename assigned to ``split`` across all class folders."""
    result: set[str] = set()
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        siblings = [p.name for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
        result |= split_for_class(siblings)[split]
    return result


def main() -> int:
    """Write ``aid-train.txt``, ``aid-val.txt``, and ``aid-test.txt``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/aid/AID"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        names = sorted(basenames_for_split(args.root, split))
        out_path = args.output_dir / f"aid-{split}.txt"
        out_path.write_text("\n".join(names) + "\n")
        logger.info("%s: %d -> %s", split, len(names), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
