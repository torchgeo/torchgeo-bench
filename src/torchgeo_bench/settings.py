"""Static defaults and config plumbing for the run/flops pipelines.

Each settings group is a real ``@dataclass`` (so a typo in
a field name is a ``TypeError``/``AttributeError``, not a silently-ignored
dict key), :func:`merge` implements the one merge rule the config system
uses ("dataclasses/dicts merge per-field, lists and scalars replace"), and
:func:`load_yaml` reads YAML with a float resolver fixed so bare-exponent
scalars like ``1e-3`` parse as floats instead of silently staying strings (a
well-known PyYAML gap).

Model configs (``conf/model/*.yaml``) are intentionally kept as plain
``dict``: they are heterogeneous, arbitrary ``_target_``-plus-kwargs blobs
passed straight to a model constructor, not a fixed schema this codebase
defines -- a dataclass would either have to special-case every model
wrapper's constructor arguments or fall back to **kwargs, which is exactly
the "magic" this design avoids elsewhere.
"""

import copy
import re
from dataclasses import asdict as dc_asdict
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import yaml

from torchgeo_bench.results import (
    DEFAULT_INTRINSIC_DIM_RESULTS_DIR,
    DEFAULT_PROFILE_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
)


def merge(base: Any, overrides: dict) -> Any:
    """Merge ``overrides`` onto ``base``: dataclasses/dicts merge per-field/key, else replace.

    Neither input is mutated; a new instance is returned. This is the one
    merge rule the whole config system uses -- notably, a model config's
    ``eval.segmentation.layers`` merges over the global segmentation
    defaults without dropping sibling fields (``head_type``, ``criterion``,
    ...), while ``layers`` itself (a list) is replaced outright rather than
    concatenated.

    A dataclass ``base`` rejects an unknown override key immediately
    (``ValueError``): a typo in a ``--config`` YAML is a startup error, not a
    silently-ignored setting. A plain-``dict`` ``base`` (model configs) has
    no fixed schema, so any key is accepted.
    """
    if not overrides:
        return base
    if is_dataclass(base) and not isinstance(base, type):
        valid = {f.name for f in dc_fields(base)}
        updates: dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in valid:
                raise ValueError(
                    f"Unknown setting {key!r} for {type(base).__name__}; "
                    f"expected one of {sorted(valid)}."
                )
            current = getattr(base, key)
            updates[key] = merge(current, value) if isinstance(value, dict) else value
        return dc_replace(base, **updates)
    if isinstance(base, dict):
        result = dict(base)
        for key, value in overrides.items():
            current = result.get(key)
            if isinstance(value, dict) and isinstance(current, dict):
                result[key] = merge(current, value)
            else:
                result[key] = value
        return result
    return overrides


def to_dict(value: Any) -> Any:
    """Recursively convert a settings dataclass (or plain dict/model config) to plain dict.

    Used wherever downstream code needs JSON-native values: ``_target_``
    instantiation kwargs, ``--print-config`` YAML output, and the resume
    config-hash payload. A dataclass field's own nested dataclasses/dicts/
    lists are handled by ``dataclasses.asdict``; a plain ``dict`` (model
    configs) is deep-copied so callers can freely mutate the result.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return dc_asdict(value)
    if isinstance(value, dict):
        return copy.deepcopy(value)
    return value


# PyYAML's default SafeLoader float resolver requires a decimal point, so a
# bare-exponent scalar like `1e-3` resolves as the *string* "1e-3" rather than
# the float 0.001 (a longstanding PyYAML issue). This adds a fixed regex so
# config YAML behaves the way every author of a `lr: 1e-3` line expects.
class _Loader(yaml.SafeLoader):
    """``SafeLoader`` with a float resolver that accepts bare exponents."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
         [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file with the bare-exponent-safe float resolver."""
    with open(path, encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_Loader)


# ---------------------------------------------------------------------------
# Nested settings groups shared by `torchgeo-bench run` and `flops`.
# ---------------------------------------------------------------------------


@dataclass
class CoordSettings:
    """CoordBench location-encoder track (used only when ``mode="coord"``)."""

    output: str = "results/coordbench_results.csv"
    names: str | list[str] = "all"  # all | comma-separated families (pdfm,satclip,...) or names
    methods: list[str] = field(default_factory=lambda: ["knn", "linear"])  # subset of {knn, linear}
    split: str = "random"  # random | spatial | both (block-CV holdout)
    folds: int = 5
    cell_deg: float = 10.0  # spatial-block grid-cell size in degrees
    knn_k: int = 5
    knn_device: str = "cpu"  # faiss device for KNN; encoder+linear still use `device`


@dataclass
class DatasetSettings:
    """Dataset loading settings. ``names`` is "all" or a list of dataset identifiers."""

    names: str | list[str] = "all"
    partition: str = "default"
    batch_size: int = 64
    num_workers: int = 4  # set 0 to disable dataloader multiprocessing
    normalization: str = (
        "bandspec_zscore"  # bandspec_zscore | model_native | minmax | minmax_zscore | identity
    )
    bands: str | list[str] = "rgb"  # rgb | all | list of band names (e.g., [red, green, blue, nir])
    image_size: int | None = 224  # e.g., 224; if null, no resizing
    time_steps: int | None = None  # multi-temporal datasets only (pastis): dates per sample
    interpolation: str = "bilinear"  # area | bilinear | bicubic | nearest; used if image_size set


@dataclass
class CalibrationSettings:
    """Calibration metrics (ECE / RMS-CE / MCE) reported per classification row."""

    n_bins_knn: int | None = None  # null = knn_k + 1
    n_bins_linear: int = 15
    temp_scale: bool = False  # requires merge_val=false so validation remains held out


@dataclass
class IntrinsicDimSettings:
    """Optional intrinsic-dimension (ID) + feature-spectrum diagnostics on embeddings."""

    enabled: bool = False
    estimators: list[str] = field(default_factory=lambda: ["TwoNN", "MLE", "lPCA"])
    splits: list[str] = field(default_factory=lambda: ["train"])  # any of: train, val, test
    max_samples: int | None = 10000  # null disables subsampling
    device: str | None = None  # null = auto (cuda if available, else cpu)


@dataclass
class CpuThroughputSettings:
    """Optional CPU pass -- adds *_cpu metrics so the explorer can show deployability."""

    enabled: bool = False
    batch_size: int = 8  # smaller than GPU batch -- CPUs choke at 256
    n_warmup: int = 1
    n_measure: int = 5
    time_budget_s: float = 300.0  # hard cap per (model, dataset)


@dataclass
class ProfileSettings:
    """Compute/efficiency profile metrics: throughput, peak GPU mem, latency, params, GMACs."""

    enabled: bool = False
    n_warmup: int = 3  # forward passes discarded before timing
    n_measure: int = 20  # timed forward passes
    cpu_throughput: CpuThroughputSettings = field(default_factory=CpuThroughputSettings)


@dataclass
class SegmentationSettings:
    """Segmentation probe settings, shared by the real eval and the flops pipeline."""

    head_type: str = "fpn"
    # How multi-temporal inputs (B, T, C, H, W) are reduced before the decoder:
    # the backbone encodes each date, then features are pooled over T. Only
    # applies to datasets that deliver a time axis.
    temporal_pool: str = "mean"  # mean | max
    layers: list[str] = field(default_factory=list)  # must be overridden per model
    lr: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    criterion: dict = field(
        default_factory=lambda: {"_target_": "torch.nn.CrossEntropyLoss", "ignore_index": 255}
    )
    lr_scheduler: str = "cosine"  # "cosine" (CosineAnnealingLR) or "none"
    # Feature caching: run frozen backbone once, cache features to RAM for head training.
    cache_features: bool = True  # pre-extract backbone features (large speedup for ViT)
    cache_dtype: str = "float16"  # storage dtype: "float16" (half RAM) or "float32"
    # Visualization (opt-in)
    save_viz: bool = False  # set true to save prediction grids and confusion matrices
    viz_dir: str = "viz"  # root output directory for visualizations
    n_viz_samples: int = 8  # number of test samples shown in the sample grid


@dataclass
class EvalSettings:
    """Evaluation settings for `torchgeo-bench run`."""

    bootstrap: int = 200  # number of bootstrap resamples
    c_range: list[float] = field(default_factory=lambda: [-6, 4, 40])  # log10 [start, stop, num]
    merge_val: bool = True  # merge train+val for final logistic training
    skip_linear: bool = False  # if true, skip LogisticRegression evaluation
    knn_k: int = 5  # number of neighbours for KNN evaluation
    knn_device: str | None = None  # null = inherit device, falling back to CPU if FAISS lacks GPU
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    intrinsic_dim: IntrinsicDimSettings = field(default_factory=IntrinsicDimSettings)
    profile: ProfileSettings = field(default_factory=ProfileSettings)
    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)


@dataclass
class FlopsEvalSettings:
    """The subset of ``EvalSettings`` the flops pipeline actually reads."""

    segmentation: SegmentationSettings = field(default_factory=SegmentationSettings)


# ---------------------------------------------------------------------------
# Top-level settings for `torchgeo-bench run` (formerly conf/config.yaml).
# ---------------------------------------------------------------------------


@dataclass
class RunSettings:
    """Top-level settings for ``torchgeo-bench run`` (and its ``coord`` mode)."""

    seed: int = 0
    # "auto" picks cuda if available, else cpu.
    device: str = "auto"
    # Explicit CSV path; null writes knn/linear/seg rows to
    # <results_dir>/<model name>.csv (and profile/intrinsic_dim rows to their
    # own dirs below); non-null routes ALL row types here.
    output: str | None = None
    results_dir: str = DEFAULT_RESULTS_DIR
    # Profile/intrinsic_dim rows are one-time model+hardware measurements, so
    # by default they land in their own per-model files instead of
    # results_dir, where a routine metrics rerun would rewrite/diff them
    # unnecessarily. Ignored when `output` is set explicitly (see above).
    profile_results_dir: str = DEFAULT_PROFILE_RESULTS_DIR
    intrinsic_dim_results_dir: str = DEFAULT_INTRINSIC_DIM_RESULTS_DIR
    verbose: bool = False
    resume: bool = False  # if true, skip dataset+method combinations already in output CSV
    mode: str = "image"  # image (default GeoBench pipeline) | coord (CoordBench)
    coord: CoordSettings = field(default_factory=CoordSettings)
    dataset: DatasetSettings = field(default_factory=DatasetSettings)
    eval: EvalSettings = field(default_factory=EvalSettings)
    # Selected model config, e.g. `conf/model/timm/resnet50.yaml` loaded as a
    # plain dict -- see the module docstring for why this isn't a dataclass.
    model: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level settings for `torchgeo-bench flops` (formerly conf/flops_config.yaml).
#
# Per-sample compute cost (GFLOPs) split into backbone / head / probe. There
# is no dataset axis: `_count_gflops` runs on a synthetic tensor, so backbone
# GFLOPs at a fixed input shape is identical across datasets. The axis that
# does matter is channel count, captured by two fixed band configs sourced
# from cloudsen12 (the canonical 12-band S2 set -- not so2sat, whose 12 bands
# are 10 S2 + 2 SAR).
# ---------------------------------------------------------------------------


@dataclass
class FlopsSettings:
    """Top-level settings for ``torchgeo-bench flops``."""

    # "auto" picks cuda if available, else cpu.
    device: str = "auto"
    resume: bool = True
    # Nothing here is trained and GFLOPs is deterministic, so this seeds no
    # measurement directly. It exists because model configs interpolate it --
    # conf/model/rcf.yaml carries `seed: ${seed}`, resolved explicitly by
    # `compose_config` against this top-level field. Held equal to the run
    # settings' default because RCF's features *are* its random projection: a
    # different seed would measure a different model.
    seed: int = 0
    output: str = "results/compute_cost.csv"
    band_configs: list[str] = field(default_factory=lambda: ["rgb", "s2"])  # 3ch / 12ch, cloudsen12
    band_source: str = "cloudsen12"
    normalization: str = "bandspec_zscore"
    image_size: int = 224
    # Segmentation head cells. num_classes is deliberately not an axis (see
    # the original flops_config.yaml derivation note this replaces).
    seg_head_types: list[str] = field(default_factory=lambda: ["fpn", "dpt"])
    seg_num_classes: int = 4
    # `rgb` is the joinable half: every row in results/all_segmentation_results.csv
    # is bands=rgb; `s2` is kept as a free S2-vs-RGB head-cost comparison.
    seg_band_configs: list[str] = field(default_factory=lambda: ["rgb", "s2"])
    # Base segmentation eval settings; each model config's own `eval.segmentation`
    # (notably `layers`) is merged over this, mirroring the real seg pipeline.
    eval: FlopsEvalSettings = field(default_factory=FlopsEvalSettings)
    # Probe head; mirrors eval.linear_head in the run settings. "mlp" adds a
    # Linear(D, D, bias=False) projection over the feature dim.
    probe_head: str = "linear"  # linear | mlp
    probe_num_classes: int = 10
    # measure_profile settings; matches the real eval's batch so throughput,
    # latency, peak memory and energy are comparable to a live run.
    timing_batch_size: int = 64
    n_warmup: int = 3
    n_measure: int = 20
    model: dict = field(default_factory=dict)
