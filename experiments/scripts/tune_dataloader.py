"""Compare data-loader batch sizes and worker counts for one model and dataset.

Report samples per second, peak GPU memory, and the fastest successful setting.

Usage::

    python experiments/scripts/tune_dataloader.py \\
        --model terratorch/prithvi_eo_v2_300 \\
        --dataset m-bigearthnet \\
        --bands all \\
        --batch-sizes 64,128,256,512 \\
        --num-workers 4,8,16,32

Loads only the training split from the dataset family's fixed local data root.
"""

import argparse
import logging
import time

import torch
from torch.utils.data import DataLoader, Dataset

from torchgeo_bench.bands import BandSpec
from torchgeo_bench.config.presets import NORMALIZATIONS, build_model, resolve_run_config
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.config.schema import ModelConfig
from torchgeo_bench.datasets import load_split
from torchgeo_bench.devices import resolve_device
from torchgeo_bench.models.interface import BenchModel

logger = logging.getLogger(__name__)


def _build_model(
    model_name: str,
    bands_list: list[BandSpec],
    dataset_name: str = "m-bigearthnet",
    train_dataset: Dataset | None = None,
) -> BenchModel:
    cfg, preset = resolve_run_config(
        RunConfig(model=ModelConfig(name=model_name), datasets=[dataset_name]), dataset_name
    )
    runtime_options = {}
    if preset.kwargs.get("mode") == "empirical":
        runtime_options["dataset"] = train_dataset
    return build_model(
        preset,
        bands=bands_list,
        normalization=NORMALIZATIONS[cfg.input.normalization],
        **runtime_options,
    )


def _bench(
    model: BenchModel, dl: DataLoader, device: torch.device, max_batches: int
) -> tuple[float, float, float]:
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
    p.add_argument("--bands", default="all", help="rgb, default, all, or comma-separated names")
    p.add_argument("--batch-sizes", default="64,128,256,512")
    p.add_argument("--num-workers", default="4,8,16,32")
    p.add_argument("--max-batches", type=int, default=20, help="cap per cell to stay quick")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device = resolve_device(args.device)
    bs_list = [int(x) for x in args.batch_sizes.split(",")]
    nw_list = [int(x) for x in args.num_workers.split(",")]

    selection = (
        args.bands if args.bands in ("rgb", "default", "all") else tuple(args.bands.split(","))
    )
    train = load_split(args.dataset, "train", bands=selection)
    model = (
        _build_model(args.model, list(train.bands), args.dataset, train.dataset).to(device).eval()
    )

    logger.info("Tuning %s on %s/%s (%s)", args.model, args.dataset, args.bands, device)
    logger.info("%6s %6s %12s %10s %10s", "bs", "nw", "samples/sec", "peak GB", "wall (s)")

    results = []
    for bs in bs_list:
        for nw in nw_list:
            loader = DataLoader(
                train.dataset,
                batch_size=bs,
                num_workers=nw,
                shuffle=False,
                pin_memory=device.type == "cuda",
                persistent_workers=nw > 0,
            )
            try:
                sps, peak, dt = _bench(model, loader, device, args.max_batches)
                results.append((sps, bs, nw, peak, dt))
                logger.info("%6d %6d %12.1f %10.2f %10.2f", bs, nw, sps, peak, dt)
            except (
                torch.cuda.OutOfMemoryError
            ) as exc:  # allow-except: oversized batches are expected during tuning
                torch.cuda.empty_cache()
                logger.warning("batch_size=%d num_workers=%d: %s", bs, nw, exc)

    if not results:
        raise RuntimeError("No dataloader configuration completed successfully.")
    best = max(results, key=lambda r: r[0])
    logger.info(
        "Best: batch_size=%d num_workers=%d, %.1f samples/sec (%.2f GB peak)",
        best[1],
        best[2],
        best[0],
        best[3],
    )


if __name__ == "__main__":
    main()
