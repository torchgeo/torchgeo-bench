#!/usr/bin/env python
"""Extract linear-probe probabilities for cleanlab dataset auditing.

For one classification dataset, look up the top-1 linear-probe model from
``results/all_results.csv``, rebuild that exact (model, dataset, bands,
normalization, image_size, partition) combo, fit a logistic regression at
the recorded ``best_c`` (merging train+val), and save predicted
probabilities for the train and test splits.

Outputs (one ``.npz`` per split)::

    results/cleanlab/probs/<dataset>__<model_name>_train.npz
    results/cleanlab/probs/<dataset>__<model_name>_test.npz

Each NPZ contains:

- ``indices``: (N,) int64 — sample indices into the underlying dataset
  (0..N-1 in dataloader order; deterministic since loaders use shuffle=False
  for val/test and we drop train shuffle).
- ``labels``: (N,) int64 single-label OR (N, C) float32 multi-label.
- ``probs``: (N, C) float32 — softmax probabilities (sigmoid for multi-label).
- ``classes``: (C,) int64 — class index ordering used in ``probs``.

This script is invoked once per dataset (typically as a SLURM array task).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets  # noqa: E402
from torchgeo_bench.linear import LogisticRegression  # noqa: E402
from torchgeo_bench.main import embed_split  # noqa: E402

logger = logging.getLogger("cleanlab_extract")


CONF_ROOT = REPO_ROOT / "src" / "torchgeo_bench" / "conf" / "model"


def build_name_to_config_map() -> dict[str, Path]:
    """Map ``name:`` field in each model yaml to its file path."""
    out: dict[str, Path] = {}
    for yaml_path in CONF_ROOT.rglob("*.yaml"):
        cfg = OmegaConf.load(yaml_path)
        name = cfg.get("name") if isinstance(cfg, dict) or hasattr(cfg, "get") else None
        if name is not None:
            out[str(name)] = yaml_path
    return out


VALID_NORMS = {"bandspec_zscore", "model_native", "minmax", "minmax_zscore", "identity"}


def lookup_top1(results_csv: Path, dataset: str) -> pd.Series:
    """Return the top-1 ``method=linear`` row for ``dataset``.

    Filters out rows whose ``normalization`` is no longer accepted by the
    current ``NormalizationStrategy`` enum (e.g. legacy ``raw`` rows from
    earlier runs). Without this filter, lookups can pick a stale top-1 that
    can no longer be re-instantiated.
    """
    df = pd.read_csv(results_csv)
    sub = df[
        (df["dataset"] == dataset)
        & (df["method"] == "linear")
        & (df["normalization"].isin(VALID_NORMS))
    ]
    if sub.empty:
        raise SystemExit(f"No linear-probe rows for dataset {dataset!r} in {results_csv}")
    return sub.sort_values("metric_value", ascending=False).iloc[0]


def parse_bands(value: object) -> str | list[str]:
    """Convert a CSV bands cell back to the form ``get_datasets`` accepts."""
    s = str(value)
    if s in ("rgb", "all"):
        return s
    return [b.strip() for b in s.split(",") if b.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. m-eurosat).")
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "results" / "all_results.csv",
        help="Input results CSV used to pick the top-1 model.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "results" / "cleanlab" / "probs",
        help="Output directory for prob NPZ files.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the linear probe (matches main.py default)."
    )
    parser.add_argument(
        "--c",
        type=float,
        default=None,
        help="Override regularization C; default uses best_c from CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prob files instead of skipping.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    row = lookup_top1(args.results, args.dataset)

    model_name = str(row["name"])
    bands_value = parse_bands(row["bands"])
    normalization = str(row["normalization"])
    image_size = (
        int(row["image_size"])
        if pd.notna(row["image_size"]) and str(row["image_size"]).strip() not in ("", "None", "nan")
        else None
    )
    interpolation = str(row.get("interpolation") or "bicubic")
    partition = str(row["partition"])
    best_c = float(args.c) if args.c is not None else float(row["best_c"])

    train_path = args.out / f"{args.dataset}__{model_name}_train.npz"
    test_path = args.out / f"{args.dataset}__{model_name}_test.npz"
    if train_path.exists() and test_path.exists() and not args.force:
        logger.warning("Skipping (already exist): %s, %s", train_path, test_path)
        return

    logger.info(
        "[%s] top-1 model=%s bands=%s norm=%s image_size=%s partition=%s best_c=%g",
        args.dataset,
        model_name,
        bands_value,
        normalization,
        image_size,
        partition,
        best_c,
    )

    name_map = build_name_to_config_map()
    if model_name not in name_map:
        raise SystemExit(f"No yaml found with name={model_name!r}; checked {CONF_ROOT}")
    model_cfg = OmegaConf.load(name_map[model_name])

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds_cls = get_bench_dataset_class(args.dataset)
    is_multilabel = ds_cls.multilabel

    result = get_datasets(
        dataset_name=args.dataset,
        partition_name=partition,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_val=True,
        image_size=image_size,
        interpolation=interpolation,
        bands=bands_value,
    )
    assert result is not None
    train_dataset, train_loader_shuffled, val_loader, test_loader = result
    # Rebuild an unshuffled train loader so saved indices match dataset order.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_loader_shuffled.batch_size,
        shuffle=False,
        num_workers=train_loader_shuffled.num_workers,
        pin_memory=train_loader_shuffled.pin_memory,
    )

    # Resolve BandSpecs for the model wrapper, exactly like main.py.
    bench_for_bands = ds_cls()
    if bands_value == "rgb":
        bands_resolved = tuple(bench_for_bands.rgb_bands)
    elif bands_value in ("all", None):
        bands_resolved = None
    else:
        bands_resolved = tuple(bands_value)
    bands_list = bench_for_bands.select_band_specs(bands_resolved)

    model = instantiate(
        model_cfg,
        bands=bands_list,
        normalization=normalization,
        _convert_="object",
    )
    model.to(device).eval()

    # Embed splits — same path as main.py.
    x_train, y_train = embed_split(model, train_loader, device, verbose=args.verbose)
    x_val, y_val = embed_split(model, val_loader, device, verbose=args.verbose)
    x_test, y_test = embed_split(model, test_loader, device, verbose=args.verbose)
    feature_dim = x_train.shape[1]
    logger.info(
        "[%s] embeddings: train=%s val=%s test=%s d=%d",
        args.dataset,
        x_train.shape,
        x_val.shape,
        x_test.shape,
        feature_dim,
    )

    # Free the GPU model — linear probe is small.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Fit linear probe on train+val (matches merge_val=True default).
    x_fit_np = np.concatenate([x_train, x_val], axis=0)
    y_fit_np = np.concatenate([y_train, y_val], axis=0)
    if is_multilabel:
        y_fit = torch.from_numpy(y_fit_np).float()
    else:
        y_fit = torch.from_numpy(y_fit_np).long()
    x_fit = torch.from_numpy(x_fit_np)

    clf = LogisticRegression(
        C=best_c,
        max_iter=4000,
        tol=1e-6,
        random_state=args.seed,
        device=args.device,
        multi_label=is_multilabel,
    )
    clf.fit(x_fit, y_fit)

    classes = (
        np.arange(y_train.shape[1], dtype=np.int64)
        if is_multilabel
        else np.asarray(clf.classes_, dtype=np.int64)
    )

    # In-sample probs on train (note: in-sample for cleanlab → underestimates train noise).
    train_probs = clf.predict_proba(torch.from_numpy(x_train)).astype(np.float32)
    test_probs = clf.predict_proba(torch.from_numpy(x_test)).astype(np.float32)

    np.savez_compressed(
        train_path,
        indices=np.arange(len(y_train), dtype=np.int64),
        labels=y_train,
        probs=train_probs,
        classes=classes,
        meta=np.array(
            [
                args.dataset,
                model_name,
                str(bands_value),
                normalization,
                str(image_size),
                partition,
                f"{best_c:g}",
                "train",
            ],
            dtype=object,
        ),
    )
    np.savez_compressed(
        test_path,
        indices=np.arange(len(y_test), dtype=np.int64),
        labels=y_test,
        probs=test_probs,
        classes=classes,
        meta=np.array(
            [
                args.dataset,
                model_name,
                str(bands_value),
                normalization,
                str(image_size),
                partition,
                f"{best_c:g}",
                "test",
            ],
            dtype=object,
        ),
    )
    logger.warning(
        "[%s] wrote %s (probs %s) and %s (probs %s)",
        args.dataset,
        train_path,
        train_probs.shape,
        test_path,
        test_probs.shape,
    )


if __name__ == "__main__":
    main()
