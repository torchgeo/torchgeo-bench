"""Temporary input bridge for callers awaiting the explicit CLI migration.

Only the old image runner's built-in sections are translated. No configuration
objects or constructor metadata cross the typed runtime boundary.
"""

from collections.abc import Callable, Mapping
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from .config_schema import RunConfig
from .image_hash import _hash_payload
from .presets import NORMALIZATIONS, merge_settings

if TYPE_CHECKING:
    from omegaconf import DictConfig


def resolve_model_config(model: "DictConfig", dataset: str) -> "DictConfig":
    """Keep the old recipe-only helper available until maintained callers migrate."""
    from omegaconf import DictConfig, OmegaConf

    raw = OmegaConf.to_container(model, resolve=True)
    assert isinstance(raw, dict)
    values = cast(dict[str, Any], raw)
    overrides = values.pop("dataset_overrides", {})
    return DictConfig(merge_settings(values, overrides.get(dataset, {})))


def legacy_run_config(raw: Mapping[str, Any], dataset: str) -> RunConfig:
    """Convert one old image run into its explicit, dataset-resolved settings."""
    model = dict(raw["model"])
    overrides = model.pop("dataset_overrides", {})
    model = merge_settings(model, overrides.get(dataset, {}))
    inputs = dict(raw["dataset"])
    evaluation = dict(raw["eval"])
    model_eval = model.pop("eval", {})
    segmentation = merge_settings(evaluation["segmentation"], model_eval.get("segmentation", {}))
    criterion = segmentation.pop("criterion")
    if set(criterion) - {"_target_", "ignore_index"} or criterion["_target_"] != (
        "torch.nn.CrossEntropyLoss"
    ):
        raise ValueError("The typed image runner supports CrossEntropyLoss with ignore_index")
    for old, new in (("head_type", "head"), ("lr", "learning_rate"), ("lr_scheduler", "scheduler")):
        if old in segmentation:
            segmentation[new] = segmentation.pop(old)
    for key in ("save_viz", "viz_dir", "n_viz_samples"):
        segmentation.pop(key, None)
    segmentation["ignore_index"] = criterion.get("ignore_index", 255)
    start, stop, count = model_eval.get("c_range", evaluation["c_range"])
    for key in ("image_size", "interpolation"):
        if key in model:
            inputs[key] = model.pop(key)
    name, target = model.pop("name"), model.pop("_target_")
    return RunConfig.model_validate(
        {
            "model": {"name": name, "target": target, "kwargs": model},
            "datasets": [dataset],
            "input": {
                "bands": inputs.get("bands") or "all",
                "partition": inputs["partition"],
                "normalization": {v: k for k, v in NORMALIZATIONS.items()}[inputs["normalization"]],
                "image_size": inputs.get("image_size"),
                "interpolation": inputs.get("interpolation", "bilinear"),
                "time_steps": inputs.get("time_steps"),
            },
            "runtime": {
                "device": raw["device"],
                "seed": raw["seed"],
                "verbose": raw["verbose"],
                "batch_size": inputs["batch_size"],
                "workers": inputs["num_workers"],
            },
            "classification": {
                "methods": ["knn"] if evaluation["skip_linear"] else ["knn", "linear"],
                "knn_k": model_eval.get("knn_k", evaluation["knn_k"]),
                "knn_device": evaluation["knn_device"],
                "bootstrap_samples": evaluation["bootstrap"],
                "linear": {
                    "c_log10_start": float(start),
                    "c_log10_stop": float(stop),
                    "c_count": count,
                    "refit_train_val": evaluation["merge_val"],
                },
                "calibration": evaluation["calibration"],
            },
            "segmentation": segmentation,
            "profile": evaluation.get("profile", {}),
            "intrinsic_dim": evaluation.get("intrinsic_dim", {}),
            "output": {
                "file": raw.get("output"),
                "directory": raw.get("results_dir", "results/models"),
                "profile_directory": raw.get("profile_results_dir", "results/profiles"),
                "intrinsic_dim_directory": raw.get(
                    "intrinsic_dim_results_dir", "results/intrinsic_dim"
                ),
                "resume": raw["resume"],
            },
        }
    )


def accept_legacy_config(function: Callable[..., None]) -> Callable[..., None]:
    """Adapt only the transitional main entry point; typed calls pass through unchanged."""

    @wraps(function)
    def wrapped(config: RunConfig, *, strict: bool = False) -> None:
        if isinstance(config, RunConfig):
            function(config, strict=strict)
            return
        from omegaconf import OmegaConf

        if config.get("mode", "image") == "coord":
            from .coordbench.run import run_coordbench

            run_coordbench(config)
            return
        from .datasets import list_datasets

        raw = OmegaConf.to_container(config, resolve=True)
        assert isinstance(raw, dict)
        values = cast(dict[str, Any], raw)
        config_hash = _legacy_resume_hash(values)
        names = raw["dataset"]["names"]
        if isinstance(names, str):
            names = list_datasets() if names == "all" else names.split(",")
        for dataset in names:
            function(legacy_run_config(values, dataset), strict=strict, _legacy_hash=config_hash)

    return wrapped


def _legacy_resume_hash(raw: Mapping[str, Any]) -> str:
    """Retain the original fingerprint while the bridge resolves constructor metadata."""
    dataset = {key: value for key, value in raw["dataset"].items() if key != "names"}
    evaluation = {
        key: value for key, value in raw["eval"].items() if key not in {"profile", "intrinsic_dim"}
    }
    evaluation["segmentation"] = {
        "save_viz": False,
        "viz_dir": "viz",
        "n_viz_samples": 8,
        **evaluation["segmentation"],
    }
    return _hash_payload(
        {
            "version": 1,
            "seed": raw["seed"],
            "device": raw["device"],
            "dataset": dataset,
            "eval": evaluation,
            "model": raw["model"],
        }
    )
