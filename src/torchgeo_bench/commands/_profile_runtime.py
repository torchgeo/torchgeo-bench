# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Runtime implementation for the standalone profile command."""

import hashlib
import json
import platform
import sys
from contextlib import redirect_stdout
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..config.presets import NORMALIZATIONS, ModelPreset, build_model
from ..config.profile import ProfileConfig, resolve_profile_config
from ..datasets import BandSpec, LoadedSplit, ResolvedInput, load_split, resolve_input
from ..devices import resolve_device
from ..model_profile import ProfileResult, profile_inference


def _load_batch(config: ProfileConfig) -> tuple[LoadedSplit, torch.Tensor]:
    """Load one full batch and its ordered band metadata."""
    inputs = config.input
    resolved = resolve_input(
        config.dataset, bands=inputs.bands, partition=inputs.partition, time_steps=inputs.time_steps
    )
    train = load_split(
        config.dataset,
        "train",
        image_size=inputs.image_size,
        interpolation=inputs.interpolation,
        bands=inputs.bands,
        partition=inputs.partition,
        time_steps=inputs.time_steps,
        inputs=resolved,
    )
    train_loader = DataLoader(
        train.dataset,
        batch_size=config.runtime.batch_size,
        shuffle=True,
        num_workers=config.runtime.workers,
        pin_memory=torch.cuda.is_available(),
    )
    batch = next(iter(train_loader))["image"]
    if batch.shape[0] != config.runtime.batch_size:
        raise RuntimeError(
            f"dataset returned batch size {batch.shape[0]}, requested {config.runtime.batch_size}"
        )
    expected_rank = 5 if train.input.layout == "TCHW" else 4
    if (
        batch.ndim != expected_rank
        or batch.shape[-3] != len(train.bands)
        or (expected_rank == 5 and batch.shape[1] != train.input.num_time_steps)
    ):
        raise ValueError(
            f"{train.dataset_name}: batch {tuple(batch.shape)} disagrees with "
            f"resolved {train.input.layout} input ({len(train.bands)} channels)"
        )
    return train, batch


def _construction_preset(
    preset: ModelPreset, config: ProfileConfig, sample: torch.Tensor
) -> ModelPreset:
    """Supply the input-dependent shape required by Scale-MAE's positional embedding."""
    if preset.target in {
        "torchgeo_bench.models.TorchGeoScaleMAEBench",
        "torchgeo_bench.models.torchgeo_models.TorchGeoScaleMAEBench",
    }:
        size = config.input.image_size
        kwargs = {**preset.kwargs, "image_size": sample.shape[-1] if size is None else size}
        return preset.model_copy(update={"kwargs": kwargs})
    return preset


def _build_model(
    preset: ModelPreset,
    config: ProfileConfig,
    train_dataset: Dataset,
    selected_bands: list[BandSpec],
) -> nn.Module:
    """Inject runtime band metadata and empirical filters' source dataset."""
    options: dict[str, Any] = {
        "bands": selected_bands,
        "normalization": NORMALIZATIONS[config.input.normalization],
    }
    if (
        preset.target
        in {
            "torchgeo_bench.models.RCFBench",
            "torchgeo_bench.models.rcf.RCFBench",
        }
        and preset.kwargs.get("mode") == "empirical"
    ):
        options["dataset"] = train_dataset
    return build_model(preset, **options)


def _record(
    config: ProfileConfig,
    preset: ModelPreset,
    inputs: ResolvedInput,
    sample: torch.Tensor,
    result: ProfileResult,
) -> dict[str, Any]:
    """Attach reproducibility metadata without serializing runtime objects."""
    device = sample.device
    normalization = NORMALIZATIONS[config.input.normalization]
    resolved_model_config = {
        "name": preset.name,
        "target": preset.target,
        "kwargs": preset.model_dump(mode="json")["kwargs"],
        "input": config.input.model_dump(mode="json"),
        "dataset_input": inputs.description,
    }
    model_json = json.dumps(resolved_model_config, sort_keys=True)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": config.model.name,
        "dataset": config.dataset,
        "seed": config.runtime.seed,
        "bands": list(inputs.band_names),
        "band_selection": inputs.description["selection"],
        "dataset_input_fingerprint": inputs.fingerprint,
        "normalization": normalization,
        "input_normalization": preset.kwargs.get("input_normalization", normalization),
        "dataset_partition": config.input.partition,
        "sample_sha256": hashlib.sha256(sample.detach().cpu().numpy().tobytes()).hexdigest(),
        "model_config": resolved_model_config,
        "model_config_hash": hashlib.sha256(model_json.encode()).hexdigest(),
        "device_index": device.index,
        "input_shape": list(sample.shape),
        "image_size": config.input.image_size,
        "interpolation": config.input.interpolation,
        "device": str(device),
        "hardware": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else platform.platform()
        ),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "scope": "encoder inference on one real dataset batch",
        "profile": asdict(result),
    }


def run(config: ProfileConfig) -> None:
    """Load one real dataset batch, measure it, and write one JSON record."""
    config, preset = resolve_profile_config(config)
    try:
        device = resolve_device(config.runtime.device)
    except ValueError as error:  # allow-except: report unavailable or invalid requested devices
        raise SystemExit(f"error: {error}") from error
    torch.manual_seed(config.runtime.seed)
    np.random.seed(config.runtime.seed)  # noqa: NPY002 - Dataset transforms use NumPy's global RNG.
    # Upstream dataset/model constructors may print diagnostics; stdout is one JSON record.
    with redirect_stdout(sys.stderr):
        train, batch = _load_batch(config)
        selected_bands = list(train.bands)
        preset = _construction_preset(preset, config, batch)
        model = _build_model(preset, config, train.dataset, selected_bands).to(device).eval()
        sample = batch.to(device)
        result = profile_inference(
            model,
            sample,
            device=device,
            precision=config.precision,
            n_warmup=config.warmup,
            n_measure=config.measurements,
            count_flops=config.count_flops,
        )
    print(json.dumps(_record(config, preset, train.input, sample, result), allow_nan=False))
