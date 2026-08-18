#!/usr/bin/env python
"""Instrumented smoke test for the resize x normalization sweep.

For each smoke.jobs cell, load one batch and run one frozen forward pass with
``torch.nn.functional.interpolate`` patched to record every call. Reports, per
cell: the raw dataset tile size, the shape entering the model, how many times
the image was resized (dataset transform + wrapper internals), and whether the
model_native normalizer is defined. The sweep's native cells are only valid
for models where resize_calls == 0; the 224 cells should show exactly one
dataset-side resize (a second call flags an internal wrapper re-resize).

Run on a GPU node from the repo root:
    python experiments/scripts/slurm/resize_norm/smoke_check.py \
        --jobs experiments/scripts/slurm/resize_norm/smoke.jobs \
        --out results/sweeps/resize_norm/smoke_report.csv
"""

import argparse
import csv
import pathlib
import sys
import traceback

import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_module

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
_orig_interpolate = F.interpolate


def _counting_interpolate(input, *args, **kwargs):  # noqa: A002 - torch's name
    out = _orig_interpolate(input, *args, **kwargs)
    _calls.append((tuple(input.shape), tuple(out.shape)))
    return out


def run_cell(cfg_name, norm, size, extra, dataset, bands, device):
    """Mirror main.py's dataset/model construction, one batch, one forward."""
    from torchgeo_bench.config import instantiate
    from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
    from torchgeo_bench.main import resolve_model_config

    overrides = [
        f"+model={cfg_name}",
        f"dataset.names=[{dataset}]",
        f"dataset.bands={bands if bands in ('rgb', 'all') else '[' + bands + ']'}",
        f"dataset.normalization={norm}",
        f"dataset.image_size={size}",
        "dataset.interpolation=bilinear",
        "dataset.num_workers=0",
        "dataset.batch_size=4",
        f"device={device}",
    ] + [o for o in extra if o != "-"]
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base=None):
        cfg = compose(config_name="config", overrides=overrides)

    ds_cls = get_bench_dataset_class(dataset)
    model_cfg = resolve_model_config(cfg.model, dataset)
    effective_image_size = model_cfg.get("image_size", cfg.dataset.get("image_size"))
    effective_interpolation = model_cfg.get(
        "interpolation", cfg.dataset.get("interpolation", "bilinear")
    )

    # Raw tile size: build the dataset once with no resize transform at all.
    raw_ds, *_ = get_datasets(
        dataset_name=dataset,
        partition_name=cfg.dataset.partition,
        batch_size=4,
        num_workers=0,
        return_val=True,
        image_size=None,
        interpolation="bilinear",
        bands=cfg.dataset.bands,
    )
    raw = raw_ds[0]["image"]

    train_dataset, *_ = get_datasets(
        dataset_name=dataset,
        partition_name=cfg.dataset.partition,
        batch_size=4,
        num_workers=0,
        return_val=True,
        image_size=effective_image_size,
        interpolation=effective_interpolation,
        bands=cfg.dataset.bands,
    )
    # Deterministic batch (the train loader shuffles): first four samples, so
    # feat_checksum is comparable across protocol cells of one (model, dataset).
    images = torch.stack([train_dataset[i]["image"] for i in range(4)]).to(device)

    bench = ds_cls()
    bands_resolved = (
        tuple(bench.rgb_bands)
        if cfg.dataset.bands == "rgb"
        else None
        if cfg.dataset.bands in ("all", None)
        else tuple(cfg.dataset.bands)
    )
    bands_list = bench.select_band_specs(bands_resolved)
    instantiate_kwargs: dict = {"bands": bands_list, "normalization": norm}
    if model_cfg.get("mode", None) == "empirical":
        instantiate_kwargs["dataset"] = train_dataset
    model_cfg.pop("interpolation", None)
    model = instantiate(model_cfg, **instantiate_kwargs)
    model.to(device).eval()

    _calls.clear()
    F.interpolate = _counting_interpolate
    try:
        with torch.inference_mode():
            feats = model(images)
    finally:
        F.interpolate = _orig_interpolate
    resizes = list(_calls)

    dataset_resized = tuple(raw.shape[-2:]) != tuple(images.shape[-2:])
    total = int(dataset_resized) + len(resizes)
    return {
        "raw_hw": f"{raw.shape[-2]}x{raw.shape[-1]}",
        "model_input_hw": f"{images.shape[-2]}x{images.shape[-1]}",
        "effective_image_size": str(effective_image_size),
        "internal_resizes": len(resizes),
        "internal_shapes": ";".join(f"{i[-2]}x{i[-1]}->{o[-2]}x{o[-1]}" for i, o in resizes),
        "dataset_resized": int(dataset_resized),
        "total_resizes": total,
        "feat_dim": tuple(feats.shape)[-1],
        # Checksum over the same fixed batch: two protocol cells of one
        # (model, dataset) producing identical checksums means that grid axis
        # is collapsed for this model (e.g. a wrapper ignoring the knob).
        "feat_checksum": f"{feats.double().sum().item():.6e}",
        "status": "ok",
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    bands_map = {}
    here = pathlib.Path(__file__).parent
    for row in (here / "bands_map.tsv").read_text().splitlines():
        model, dataset, bands = row.split("\t")
        bands_map[(model, dataset)] = bands

    fields = [
        "model",
        "dataset",
        "normalization",
        "image_size",
        "raw_hw",
        "model_input_hw",
        "effective_image_size",
        "internal_resizes",
        "internal_shapes",
        "dataset_resized",
        "total_resizes",
        "feat_dim",
        "feat_checksum",
        "status",
        "error",
    ]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for job in pathlib.Path(args.jobs).read_text().splitlines():
            name, cfg_name, norm, size, _seed, datasets, extra = job.split("\t")
            for dataset in datasets.split(","):
                bands = bands_map.get((name, dataset))
                base = {
                    "model": name,
                    "dataset": dataset,
                    "normalization": norm,
                    "image_size": size,
                }
                if bands is None:
                    writer.writerow(base | {"status": "skip", "error": "no bands_map entry"})
                    continue
                print(f"--- {name} {dataset} norm={norm} size={size}", flush=True)
                try:
                    result = run_cell(
                        cfg_name, norm, size, extra.split(), dataset, bands, args.device
                    )
                except Exception as exc:  # smoke test: record and keep probing cells
                    traceback.print_exc()
                    failures += 1
                    result = {"status": "fail", "error": f"{type(exc).__name__}: {exc}"}
                writer.writerow(base | result)
                fh.flush()
    print(f"done, {failures} cell failures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
