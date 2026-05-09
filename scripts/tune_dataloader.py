"""Sweep ``(batch_size, num_workers)`` for one ``(model, dataset, bands)`` combo and
report samples/sec, peak GPU memory, and the throughput-maximising config.

Usage::

    python scripts/tune_dataloader.py \\
        --model terratorch/prithvi_eo_v2_300 \\
        --dataset m-bigearthnet \\
        --bands all \\
        --root data/classification_v1.0_wds \\
        --batch-sizes 64,128,256,512 \\
        --num-workers 4,8,16,32

Designed for the post-WebDataset layout under
``data/classification_v1.0_wds/`` so the dataloader is fork-safe at any
``num_workers``.
"""

import argparse
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.datasets._v1_webdataset import GeoBenchv1Sharded


def _build_dataset(name: str, bands: str, root: Path):
    cls = get_bench_dataset_class(name)
    bench = cls()
    sel = (
        tuple(bench.rgb_bands)
        if bands == "rgb"
        else None
        if bands == "all"
        else tuple(bands.split(","))
    )
    band_names = [b.source_name for b in bench.select_band_specs(sel)]
    return GeoBenchv1Sharded(
        root=str(root),
        dataset_name=name,
        split="train",
        partition="default",
        bands=tuple(band_names),
    )


def _build_model(model_cfg: str, bands_list):
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base=None):
        cfg = compose(config_name="config", overrides=[f"model={model_cfg}"])
    return instantiate(
        cfg.model,
        bands=bands_list,
        normalization="bandspec_zscore",
        _convert_="object",
    )


def _bench(
    model, dataset, batch_size: int, num_workers: int, device: torch.device, max_batches: int
):
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(dl):
            x = batch["image"].to(device, non_blocking=True)
            _ = model.forward_patch_features(x)
            n += x.shape[0]
            if i + 1 >= max_batches:
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
    return n / dt, peak_gb, dt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="e.g. terratorch/prithvi_eo_v2_300")
    p.add_argument("--dataset", default="m-bigearthnet")
    p.add_argument("--bands", default="all")
    p.add_argument("--root", default="data/classification_v1.0_wds")
    p.add_argument("--batch-sizes", default="64,128,256,512")
    p.add_argument("--num-workers", default="4,8,16,32")
    p.add_argument("--max-batches", type=int, default=20, help="cap per cell to stay quick")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bs_list = [int(x) for x in args.batch_sizes.split(",")]
    nw_list = [int(x) for x in args.num_workers.split(",")]

    dataset = _build_dataset(args.dataset, args.bands, Path(args.root))
    bench_cls = get_bench_dataset_class(args.dataset)()
    sel = (
        tuple(bench_cls.rgb_bands)
        if args.bands == "rgb"
        else None
        if args.bands == "all"
        else tuple(args.bands.split(","))
    )
    bands_list = bench_cls.select_band_specs(sel)
    model = _build_model(args.model, bands_list).to(device).eval()

    print(f"\nTuning {args.model} × {args.dataset}/{args.bands} on {device}")
    print(f"{'bs':>5} {'nw':>4} {'samples/sec':>14} {'peak GB':>10} {'wall':>8}")
    results = []
    for bs in bs_list:
        for nw in nw_list:
            try:
                sps, peak, dt = _bench(model, dataset, bs, nw, device, args.max_batches)
                results.append((sps, bs, nw, peak, dt))
                print(f"{bs:>5} {nw:>4} {sps:>14.1f} {peak:>10.2f} {dt:>7.2f}s")
            except Exception as e:
                print(f"{bs:>5} {nw:>4} FAIL  {type(e).__name__}: {str(e)[:60]}")

    if results:
        best = max(results, key=lambda r: r[0])
        print(
            f"\nBEST: batch_size={best[1]} num_workers={best[2]} "
            f"-> {best[0]:.1f} samples/sec ({best[3]:.2f} GB peak)"
        )


if __name__ == "__main__":
    main()
