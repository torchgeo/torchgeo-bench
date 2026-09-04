# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Runtime implementation for the standalone profile command."""

import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime

import torch

from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.model_profile import profile_inference
from torchgeo_bench.utils import resolve_device


def run(args) -> None:
    """Load one real dataset batch, measure it, and write one JSON record."""
    if args.batch_size <= 0 or args.warmup < 0 or args.measurements <= 0:
        raise SystemExit(
            "error: batch-size must be positive, warmup non-negative, measurements positive"
        )
    requested = resolve_device(args.device)
    if args.device.startswith("cuda") and requested.type != "cuda":
        raise SystemExit(f"error: requested {args.device!r}, but CUDA is unavailable")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"error: requested {args.device!r}, but CUDA is unavailable")

    dataset_class = get_bench_dataset_class(args.dataset)
    bands = (
        args.bands
        if args.bands in {"rgb", "all"}
        else [name.strip() for name in args.bands.split(",")]
    )
    _, train_loader, _, _ = get_datasets(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
        return_val=True,
        image_size=args.image_size,
        bands=bands,
    )
    batch = next(iter(train_loader))["image"]
    if batch.shape[0] != args.batch_size:
        raise RuntimeError(
            f"dataset returned batch size {batch.shape[0]}, requested {args.batch_size}"
        )
    dataset = dataset_class()
    selected_bands = dataset.select_band_specs(
        tuple(dataset.rgb_bands)
        if args.bands == "rgb"
        else None
        if args.bands == "all"
        else tuple(bands)
    )
    cfg = compose_config([f"model={args.model}"])
    model_cfg = dict(cfg.model)
    model_cfg.pop("eval", None)
    model = instantiate(model_cfg, bands=selected_bands, normalization="bandspec_zscore")
    model = model.to(requested).eval()
    sample = batch.to(requested)
    result = profile_inference(
        model,
        sample,
        device=requested,
        precision=args.precision,
        n_warmup=args.warmup,
        n_measure=args.measurements,
        count_flops=args.count_flops,
    )
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": args.model,
        "dataset": args.dataset,
        "bands": [spec.name for spec in selected_bands],
        "input_shape": list(sample.shape),
        "device": str(requested),
        "hardware": platform.platform(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "scope": "encoder inference on one real dataset batch",
        "profile": asdict(result),
    }
    print(json.dumps(record, allow_nan=False, default=lambda value: value.asdict()))
