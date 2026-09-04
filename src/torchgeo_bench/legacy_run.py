# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Compatibility adapter from the R01 schema to the legacy runner."""

import torch
from omegaconf import open_dict

from torchgeo_bench.config import compose_config
from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.main import main


def run(config: RunConfig) -> None:  # noqa: C901, PLR0915 - explicit schema-to-legacy mapping
    """Execute a validated image config through the legacy runner."""
    device = config.runtime.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device {device!r} requested but CUDA is unavailable')

    legacy = compose_config([f'model={config.model.name}'])
    # Keep model-specific eval defaults, then apply schema-owned settings below.
    explicit_input = config.input.model_fields_set
    with open_dict(legacy.model):
        if 'image_size' in explicit_input:
            legacy.model.pop('image_size', None)
        if 'interpolation' in explicit_input:
            legacy.model.pop('interpolation', None)
        dataset_overrides = legacy.model.get('dataset_overrides', {})
        for override in dataset_overrides.values():
            if 'image_size' in explicit_input:
                override.pop('image_size', None)
            if 'interpolation' in explicit_input:
                override.pop('interpolation', None)
    legacy.seed = config.runtime.seed
    legacy.device = device
    legacy.verbose = config.runtime.verbose
    legacy.resume = config.output.resume
    legacy.output = config.output.file
    legacy.results_dir = config.output.directory
    legacy.dataset.names = config.datasets
    legacy.dataset.partition = config.input.partition
    legacy.dataset.batch_size = config.runtime.batch_size
    legacy.dataset.num_workers = config.runtime.workers
    legacy.dataset.bands = config.input.bands
    if 'image_size' in explicit_input:
        legacy.dataset.image_size = config.input.image_size
    if 'time_steps' in explicit_input:
        legacy.dataset.time_steps = config.input.time_steps
    if 'interpolation' in explicit_input:
        legacy.dataset.interpolation = config.input.interpolation
    legacy.dataset.normalization = {
        'dataset': 'bandspec_zscore',
        'model': 'model_native',
        'minmax': 'minmax',
        'none': 'identity',
    }[config.input.normalization]
    legacy.eval.skip_linear = 'linear' not in config.classification.methods
    legacy.eval.knn_k = config.classification.knn_k
    legacy.eval.knn_device = config.classification.knn_device
    legacy.eval.bootstrap = config.classification.bootstrap_samples
    legacy.eval.merge_val = config.classification.linear.refit_train_val
    legacy.eval.c_range = [
        config.classification.linear.c_log10_start,
        config.classification.linear.c_log10_stop,
        config.classification.linear.c_count,
    ]
    legacy.eval.calibration.temp_scale = config.classification.calibration.temp_scale
    legacy.eval.calibration.n_bins_knn = config.classification.calibration.n_bins_knn
    legacy.eval.calibration.n_bins_linear = (
        config.classification.calibration.n_bins_linear
    )
    legacy.eval.segmentation.head_type = config.segmentation.head
    if 'layers' in config.segmentation.model_fields_set:
        legacy.eval.segmentation.layers = config.segmentation.layers
    elif 'eval' in legacy.model and 'segmentation' in legacy.model.eval:
        legacy.eval.segmentation.layers = legacy.model.eval.segmentation.layers
    legacy.eval.segmentation.lr = config.segmentation.learning_rate
    legacy.eval.segmentation.epochs = config.segmentation.epochs
    legacy.eval.segmentation.batch_size = config.segmentation.batch_size
    legacy.eval.segmentation.temporal_pool = config.segmentation.temporal_pool
    legacy.eval.segmentation.lr_scheduler = config.segmentation.scheduler
    legacy.eval.segmentation.criterion = {
        '_target_': 'torch.nn.CrossEntropyLoss',
        'ignore_index': config.segmentation.ignore_index,
    }
    legacy.eval.segmentation.cache_features = config.segmentation.cache_features
    legacy.eval.segmentation.cache_dtype = config.segmentation.cache_dtype
    main(legacy, strict=True)
