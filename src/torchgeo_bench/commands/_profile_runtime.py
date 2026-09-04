# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Runtime implementation for the standalone profile command."""

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np
import torch
from omegaconf import OmegaConf

from ..config import compose_config, instantiate, list_model_configs
from ..datasets import get_bench_dataset_class, get_datasets, list_datasets
from ..main import resolve_model_config
from ..model_profile import profile_inference
from ..utils import resolve_device


def run(args: argparse.Namespace) -> None:
    """Load one real dataset batch, measure it, and write one JSON record."""
    if args.batch_size <= 0 or args.warmup < 0 or args.measurements <= 0:
        raise SystemExit(
            'error: batch-size must be positive, warmup non-negative, measurements positive'
        )
    if args.model not in list_model_configs():
        raise SystemExit(f'error: unknown model {args.model!r}')
    if args.dataset not in list_datasets():
        raise SystemExit(f'error: unknown dataset {args.dataset!r}')
    if args.seed < 0 or args.image_size <= 0:
        raise SystemExit('error: seed must be non-negative and image-size positive')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit(f'error: requested {args.device!r}, but CUDA is unavailable')
    requested = resolve_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset_class = get_bench_dataset_class(args.dataset)
    bands = (
        args.bands
        if args.bands in {'rgb', 'all'}
        else [name.strip() for name in args.bands.split(',')]
    )
    if isinstance(bands, list) and (not bands or any(not name for name in bands)):
        raise SystemExit('error: bands must be non-empty names')
    _, train_loader, _, _ = get_datasets(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=0,
        return_val=True,
        image_size=args.image_size,
        bands=bands,
        partition_name=args.partition,
    )
    batch = next(iter(train_loader))['image']
    if batch.shape[0] != args.batch_size:
        raise RuntimeError(
            f'dataset returned batch size {batch.shape[0]}, requested {args.batch_size}'
        )
    dataset = dataset_class()
    selected_bands = dataset.select_band_specs(
        tuple(dataset.rgb_bands)
        if args.bands == 'rgb'
        else None
        if args.bands == 'all'
        else tuple(bands)
    )
    cfg = compose_config([f'model={args.model}'])
    model_cfg = resolve_model_config(cfg.model, args.dataset)
    resolved_model_config = OmegaConf.to_container(model_cfg, resolve=True)
    if 'eval' in model_cfg:
        model_cfg.eval = {}
    model = instantiate(
        model_cfg, bands=selected_bands, normalization=args.normalization
    )
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
    sample_hash = hashlib.sha256(sample.detach().cpu().numpy().tobytes()).hexdigest()
    model_config_json = json.dumps(resolved_model_config, sort_keys=True, default=str)
    model_config_hash = hashlib.sha256(model_config_json.encode()).hexdigest()
    device_index = (
        requested.index
        if requested.index is not None
        else torch.cuda.current_device()
        if requested.type == 'cuda'
        else None
    )
    hardware = (
        torch.cuda.get_device_name(device_index)
        if requested.type == 'cuda'
        else platform.platform()
    )
    record = {
        'timestamp_utc': datetime.now(UTC).isoformat(),
        'model': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'bands': [spec.name for spec in selected_bands],
        'normalization': args.normalization,
        'input_normalization': args.normalization,
        'dataset_partition': args.partition,
        'sample_sha256': sample_hash,
        'model_config': resolved_model_config,
        'model_config_hash': model_config_hash,
        'device_index': device_index,
        'input_shape': list(sample.shape),
        'device': str(requested),
        'hardware': hardware,
        'torch_version': torch.__version__,
        'python_version': sys.version.split()[0],
        'scope': 'encoder inference on one real dataset batch',
        'profile': asdict(result),
    }
    print(json.dumps(record, allow_nan=False))
