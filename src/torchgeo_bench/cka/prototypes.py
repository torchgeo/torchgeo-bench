"""Shared data-preparation helpers for the GeoFM CKA visualization prototypes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_MODELS = [
    "tt_clay_v1_5_base",
    "tt_prithvi_eo_v2_300_tl",
    "tgeo_dofa_base",
    "olmoearth_v1_1_base",
    "olmoearth_v1_1_tiny",
    "tt_terramind_v1_base_rgb",
    "tgeo_panopticon",
    "vit_large_patch16_dinov3sat",
    "convnext_large_dinov3",
    "resnet18",
    "resnet50",
    "vit_base_patch16_224",
    "vit_large_patch16_224",
    "swin_tiny_patch4_window7_224",
    "mobilenetv3_large_100",
]
CANONICAL_DATASETS = ["advance", "m-eurosat", "m-pv4ger", "resisc45", "so2sat"]

GROUP_BY_MODEL = {
    "tt_clay_v1_5_base": "EO-MAE",
    "tt_prithvi_eo_v2_300_tl": "EO-MAE",
    "tgeo_dofa_base": "EO-MAE",
    "olmoearth_v1_1_base": "EO-MAE",
    "olmoearth_v1_1_tiny": "EO-MAE",
    "tt_terramind_v1_base_rgb": "EO-MAE",
    "tgeo_panopticon": "EO-DINO",
    "vit_large_patch16_dinov3sat": "EO-DINO",
    "convnext_large_dinov3": "Nat-DINO",
    "resnet18": "Nat-sup",
    "resnet50": "Nat-sup",
    "vit_base_patch16_224": "Nat-sup",
    "vit_large_patch16_224": "Nat-sup",
    "swin_tiny_patch4_window7_224": "Nat-sup",
    "mobilenetv3_large_100": "Nat-sup",
}
SHORT_NAME = {
    "tt_clay_v1_5_base": "clay",
    "tt_prithvi_eo_v2_300_tl": "prithvi",
    "tgeo_dofa_base": "dofa",
    "olmoearth_v1_1_base": "olmo-b",
    "olmoearth_v1_1_tiny": "olmo-t",
    "tt_terramind_v1_base_rgb": "terramind",
    "tgeo_panopticon": "panopticon",
    "vit_large_patch16_dinov3sat": "dinov3sat",
    "convnext_large_dinov3": "convnext-d3",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "vit_base_patch16_224": "vit-b",
    "vit_large_patch16_224": "vit-l",
    "swin_tiny_patch4_window7_224": "swin-t",
    "mobilenetv3_large_100": "mobilenet",
}
MARKER_BY_MODEL = {
    "tt_clay_v1_5_base": "o",
    "tt_prithvi_eo_v2_300_tl": "s",
    "tgeo_dofa_base": "^",
    "olmoearth_v1_1_base": "v",
    "olmoearth_v1_1_tiny": "<",
    "tt_terramind_v1_base_rgb": ">",
    "tgeo_panopticon": "D",
    "vit_large_patch16_dinov3sat": "p",
    "convnext_large_dinov3": "h",
    "resnet18": "*",
    "resnet50": "P",
    "vit_base_patch16_224": "X",
    "vit_large_patch16_224": "8",
    "swin_tiny_patch4_window7_224": "H",
    "mobilenetv3_large_100": "d",
}
GROUP_COLORS = {
    "EO-MAE": "#E69F00",
    "EO-DINO": "#0072B2",
    "Nat-DINO": "#009E73",
    "Nat-sup": "#999999",
}
FAMILY_COLORS = {
    "EO-pretrained": "#C44E52",
    "ImageNet-pretrained": "#4C72B0",
}
GROUP_ORDER = ["EO-MAE", "EO-DINO", "Nat-DINO", "Nat-sup"]
FAMILY_ORDER = ["EO-pretrained", "ImageNet-pretrained"]
EO_GROUPS = frozenset({"EO-MAE", "EO-DINO"})
IMAGENET_GROUPS = frozenset({"Nat-DINO", "Nat-sup"})


@dataclass(frozen=True)
class ExemplarChoice:
    """Representative EO/ImageNet model pair for the mechanism panels."""

    dataset: str
    eo_model: str
    imagenet_model: str
    dataset_score: float
    eo_distance: float
    imagenet_distance: float


def group_for_model(name: str) -> str:
    """Return the four-way pretraining group for a model name."""
    return GROUP_BY_MODEL.get(name, "Other")


def family_for_model(name: str) -> str:
    """Return the two-family label for a model name."""
    group = group_for_model(name)
    if group in EO_GROUPS:
        return "EO-pretrained"
    if group in IMAGENET_GROUPS:
        return "ImageNet-pretrained"
    return "Other"


def short_name(name: str) -> str:
    """Return a compact label for a model name."""
    return SHORT_NAME.get(name, name)


def marker_for_model(name: str) -> str:
    """Return a stable marker for a model name."""
    return MARKER_BY_MODEL.get(name, "o")


def ordered_datasets(names: list[str]) -> list[str]:
    """Return dataset names in canonical study order when possible."""
    canonical = [name for name in CANONICAL_DATASETS if name in names]
    extras = sorted(name for name in names if name not in CANONICAL_DATASETS)
    return canonical + extras


def ordered_models(names: list[str]) -> list[str]:
    """Return model names in canonical study order when possible."""
    canonical = [name for name in CANONICAL_MODELS if name in names]
    extras = sorted(name for name in names if name not in CANONICAL_MODELS)
    return canonical + extras


def add_model_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Attach group, family, short-name, and marker metadata."""
    out = df.copy()
    out["group"] = out["name"].map(group_for_model)
    out["family"] = out["name"].map(family_for_model)
    out["short_name"] = out["name"].map(short_name)
    out["marker"] = out["name"].map(marker_for_model)
    return out


def resolve_scope(
    df: pd.DataFrame,
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve a usable study scope against what is actually present on disk."""
    available_models = sorted(df["name"].dropna().astype(str).unique().tolist())
    available_datasets = sorted(df["dataset"].dropna().astype(str).unique().tolist())

    if models is None:
        scoped_models = ordered_models([name for name in available_models if name in CANONICAL_MODELS])
        if not scoped_models:
            scoped_models = ordered_models(available_models)
    else:
        scoped_models = [name for name in models if name in available_models]

    if datasets is None:
        scoped_datasets = ordered_datasets([name for name in available_datasets if name in CANONICAL_DATASETS])
        if not scoped_datasets:
            scoped_datasets = ordered_datasets(available_datasets)
    else:
        scoped_datasets = [name for name in datasets if name in available_datasets]

    return scoped_models, scoped_datasets


def load_head_rows(
    cka_csv: str | Path,
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Load head rows for the current scope from ``cka_results.csv``."""
    df = pd.read_csv(cka_csv)
    scoped_models, scoped_datasets = resolve_scope(df, models=models, datasets=datasets)
    head = df[
        (df["layer_name"] == "head")
        & (df["name"].isin(scoped_models))
        & (df["dataset"].isin(scoped_datasets))
    ].copy()
    return add_model_metadata(head)


def aggregate_model_dataset_metrics(head_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate corrupted head rows to one row per ``(model, dataset)`` cell."""
    corrupted = head_df[head_df["corruption_type"] != "clean"].copy()
    grouped = (
        corrupted.groupby(["name", "dataset"], dropna=False)[
            ["cka", "spearman_drift_confidence", "frac_overconfident_high_drift"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "cka": "mean_cka",
                "spearman_drift_confidence": "mean_coupling",
                "frac_overconfident_high_drift": "mean_overconfident_high_drift",
            }
        )
    )
    nonzero = (
        corrupted.groupby(["name", "dataset"], dropna=False)["frac_overconfident_high_drift"]
        .apply(lambda s: float(np.mean(np.asarray(s, dtype=float) > 0.0)))
        .reset_index(name="share_nonzero_overconfident")
    )
    return add_model_metadata(grouped.merge(nonzero, on=["name", "dataset"], how="left"))


def aggregate_severity_metrics(head_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate corrupted head rows to one row per ``(model, dataset, severity)``."""
    corrupted = head_df[head_df["corruption_type"] != "clean"].copy()
    grouped = (
        corrupted.groupby(["name", "dataset", "severity"], dropna=False)[
            ["cka", "frac_overconfident_high_drift"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "cka": "mean_cka",
                "frac_overconfident_high_drift": "mean_overconfident_high_drift",
            }
        )
    )
    return add_model_metadata(grouped)


def aggregate_condition_slice_metrics(
    head_df: pd.DataFrame,
    *,
    severities: list[int],
) -> pd.DataFrame:
    """Aggregate selected clean/corrupted slices to one row per condition cell."""
    frames: list[pd.DataFrame] = []
    for severity in severities:
        if severity == 0:
            subset = head_df[head_df["corruption_type"] == "clean"].copy()
        else:
            subset = head_df[
                (head_df["corruption_type"] != "clean") & (head_df["severity"] == severity)
            ].copy()
        if subset.empty:
            continue
        grouped = (
            subset.groupby(["name", "dataset"], dropna=False)[
                ["cka", "frac_overconfident_high_drift"]
            ]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "cka": "mean_cka",
                    "frac_overconfident_high_drift": "mean_overconfident_high_drift",
                }
            )
        )
        grouped["severity"] = severity
        frames.append(grouped)
    if not frames:
        return add_model_metadata(pd.DataFrame(columns=["name", "dataset", "severity"]))
    return add_model_metadata(pd.concat(frames, ignore_index=True))


def aggregate_condition_slice_by_corruption_metrics(
    head_df: pd.DataFrame,
    *,
    severities: list[int],
) -> pd.DataFrame:
    """Aggregate corrupted rows to one row per ``(model, dataset, corruption, severity)``."""
    subset = head_df[
        (head_df["corruption_type"] != "clean") & (head_df["severity"].isin(severities))
    ].copy()
    grouped = (
        subset.groupby(["name", "dataset", "corruption_type", "severity"], dropna=False)["cka"]
        .mean()
        .reset_index()
        .rename(columns={"cka": "mean_cka"})
    )
    return add_model_metadata(grouped)


def aggregate_severity_by_corruption(head_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate corrupted head rows to one row per condition."""
    corrupted = head_df[head_df["corruption_type"] != "clean"].copy()
    grouped = (
        corrupted.groupby(["name", "dataset", "corruption_type", "severity"], dropna=False)[
            ["cka", "frac_overconfident_high_drift"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "cka": "mean_cka",
                "frac_overconfident_high_drift": "mean_overconfident_high_drift",
            }
        )
    )
    return add_model_metadata(grouped)


def load_uq_metric_rows(
    uq_csv: str | Path,
    *,
    metric_name: str,
    uq_method: str = "uncalibrated",
    models: list[str] | None = None,
    datasets: list[str] | None = None,
) -> pd.DataFrame:
    """Load scoped UQ rows for a single metric and method."""
    df = pd.read_csv(uq_csv, low_memory=False)
    scoped_models, scoped_datasets = resolve_scope(df, models=models, datasets=datasets)
    rows = df[
        (df["name"].isin(scoped_models))
        & (df["dataset"].isin(scoped_datasets))
        & (df["uq_method"] == uq_method)
        & (df["metric_name"] == metric_name)
    ].copy()
    return add_model_metadata(rows)


def aggregate_uq_metric_slices(
    uq_df: pd.DataFrame,
    *,
    severities: list[int],
    value_column: str = "metric_value",
    output_column: str = "mean_metric_value",
) -> pd.DataFrame:
    """Aggregate a UQ metric to one row per ``(model, dataset, severity)`` slice."""
    subset = uq_df[uq_df["severity"].isin(severities)].copy()
    grouped = (
        subset.groupby(["name", "dataset", "severity"], dropna=False)[value_column]
        .mean()
        .reset_index()
        .rename(columns={value_column: output_column})
    )
    return add_model_metadata(grouped)


def aggregate_uq_metric_slices_by_corruption(
    uq_df: pd.DataFrame,
    *,
    severities: list[int],
    value_column: str = "metric_value",
    output_column: str = "mean_metric_value",
) -> pd.DataFrame:
    """Aggregate a UQ metric to one row per ``(model, dataset, corruption, severity)``."""
    subset = uq_df[
        (uq_df["corruption_type"] != "clean") & (uq_df["severity"].isin(severities))
    ].copy()
    grouped = (
        subset.groupby(["name", "dataset", "corruption_type", "severity"], dropna=False)[value_column]
        .mean()
        .reset_index()
        .rename(columns={value_column: output_column})
    )
    return add_model_metadata(grouped)


def load_trace_table(
    traces_root: str | Path,
    *,
    models: list[str],
    datasets: list[str],
) -> pd.DataFrame:
    """Load sample-level trace parquet files for the requested scope."""
    root = Path(traces_root)
    frames: list[pd.DataFrame] = []
    for name in models:
        for dataset in datasets:
            parquet_path = root / name / f"{dataset}.parquet"
            if not parquet_path.exists():
                continue
            frame = pd.read_parquet(parquet_path).copy()
            frame["name"] = name
            frame["dataset"] = dataset
            frames.append(frame)

    if not frames:
        return pd.DataFrame(
            columns=[
                "corruption_type",
                "severity",
                "sample_idx",
                "drift",
                "confidence",
                "correct",
                "y_true",
                "y_pred",
                "logits",
                "name",
                "dataset",
                "group",
                "family",
                "short_name",
                "marker",
            ]
        )
    return add_model_metadata(pd.concat(frames, ignore_index=True))


def aggregate_wrong_confidence_slices(
    trace_df: pd.DataFrame,
    *,
    severities: list[int],
    confidence_column: str = "confidence",
    output_column: str = "mean_wrong_confidence",
) -> pd.DataFrame:
    """Aggregate mean confidence on wrong predictions by severity slice."""
    wrong = trace_df[
        trace_df["severity"].isin(severities) & (~trace_df["correct"].astype(bool))
    ].copy()
    grouped = (
        wrong.groupby(["name", "dataset", "severity"], dropna=False)[confidence_column]
        .mean()
        .reset_index()
        .rename(columns={confidence_column: output_column})
    )
    return add_model_metadata(grouped)


def assign_quadrants(
    trace_df: pd.DataFrame,
    *,
    threshold_group_cols: list[str],
    confidence_threshold: float = 0.9,
) -> pd.DataFrame:
    """Attach drift/confidence threshold columns and quadrant labels."""
    if trace_df.empty:
        out = trace_df.copy()
        out["drift_threshold"] = pd.Series(dtype=float)
        out["high_drift"] = pd.Series(dtype=bool)
        out["high_confidence"] = pd.Series(dtype=bool)
        out["dangerous"] = pd.Series(dtype=bool)
        out["quadrant"] = pd.Series(dtype=object)
        return out

    out = trace_df.copy()
    out["drift_threshold"] = out.groupby(threshold_group_cols)["drift"].transform("median")
    out["high_drift"] = out["drift"] > out["drift_threshold"]
    out["high_confidence"] = out["confidence"] > float(confidence_threshold)
    out["dangerous"] = out["high_drift"] & out["high_confidence"]
    quadrant_map = {
        (False, False): "low drift / low confidence",
        (False, True): "low drift / high confidence",
        (True, False): "high drift / low confidence",
        (True, True): "high drift / high confidence",
    }
    out["quadrant"] = [
        quadrant_map[(bool(drift), bool(conf))]
        for drift, conf in zip(out["high_drift"], out["high_confidence"], strict=True)
    ]
    return out


def summarize_quadrant_shares(
    trace_df: pd.DataFrame,
    *,
    threshold_group_cols: list[str],
    confidence_threshold: float = 0.9,
) -> pd.DataFrame:
    """Summarize quadrant occupancy by ``(model, dataset)`` for all and wrong-only samples."""
    quadrant_df = assign_quadrants(
        trace_df,
        threshold_group_cols=threshold_group_cols,
        confidence_threshold=confidence_threshold,
    )
    if quadrant_df.empty:
        return pd.DataFrame(
            columns=["name", "dataset", "family", "group", "population", "quadrant", "share"]
        )

    records: list[dict[str, object]] = []
    for (name, dataset), group in quadrant_df.groupby(["name", "dataset"], dropna=False):
        for population, subset in [
            ("all", group),
            ("wrong-only", group[~group["correct"].astype(bool)]),
        ]:
            if subset.empty:
                continue
            share_by_quadrant = subset["quadrant"].value_counts(normalize=True)
            for quadrant, share in share_by_quadrant.items():
                records.append(
                    {
                        "name": name,
                        "dataset": dataset,
                        "group": group_for_model(str(name)),
                        "family": family_for_model(str(name)),
                        "population": population,
                        "quadrant": quadrant,
                        "share": float(share),
                    }
                )
    return pd.DataFrame.from_records(records)


def available_trace_models(
    traces_root: str | Path,
    *,
    dataset: str,
    models: list[str],
) -> list[str]:
    """Return models that have a trace parquet for the given dataset."""
    root = Path(traces_root)
    return [
        name
        for name in models
        if (root / name / f"{dataset}.parquet").exists()
    ]


def select_exemplar(
    cell_df: pd.DataFrame,
    *,
    traces_root: str | Path,
) -> ExemplarChoice:
    """Select a representative dataset and EO/ImageNet model pair."""
    available = cell_df.copy()
    dataset_scores: list[tuple[str, float]] = []
    global_std = available[["mean_cka", "mean_overconfident_high_drift"]].std(ddof=0)
    global_std = global_std.replace(0.0, 1.0).fillna(1.0)

    for dataset, group in available.groupby("dataset", dropna=False):
        families = set(group["family"])
        if "EO-pretrained" not in families or "ImageNet-pretrained" not in families:
            continue
        present_models = available_trace_models(
            traces_root, dataset=str(dataset), models=group["name"].dropna().astype(str).unique().tolist()
        )
        if not any(family_for_model(name) == "EO-pretrained" for name in present_models):
            continue
        if not any(family_for_model(name) == "ImageNet-pretrained" for name in present_models):
            continue

        centroids = (
            group.groupby("family")[["mean_cka", "mean_overconfident_high_drift"]]
            .mean()
            .loc[FAMILY_ORDER]
        )
        diff = (centroids.loc["EO-pretrained"] - centroids.loc["ImageNet-pretrained"]) / global_std
        dataset_scores.append((str(dataset), float(np.linalg.norm(diff.to_numpy(dtype=float)))))

    if not dataset_scores:
        raise ValueError("Could not find a dataset with both EO and ImageNet traces for exemplar selection.")

    dataset, dataset_score = max(dataset_scores, key=lambda item: item[1])
    dataset_cells = available[available["dataset"] == dataset].copy()
    dataset_cells = dataset_cells[
        dataset_cells["name"].isin(
            available_trace_models(
                traces_root,
                dataset=dataset,
                models=dataset_cells["name"].dropna().astype(str).unique().tolist(),
            )
        )
    ].copy()

    centroids = dataset_cells.groupby("family")[["mean_cka", "mean_overconfident_high_drift"]].mean()
    scales = dataset_cells[["mean_cka", "mean_overconfident_high_drift"]].std(ddof=0)
    scales = scales.replace(0.0, 1.0).fillna(1.0)

    def _pick_representative(family: str) -> tuple[str, float]:
        family_cells = dataset_cells[dataset_cells["family"] == family].copy()
        if family_cells.empty:
            raise ValueError(f"Dataset {dataset!r} does not contain family {family!r}.")
        deltas = (
            family_cells[["mean_cka", "mean_overconfident_high_drift"]] - centroids.loc[family]
        ) / scales
        distances = np.linalg.norm(deltas.to_numpy(dtype=float), axis=1)
        family_cells = family_cells.assign(_distance=distances)
        best = family_cells.sort_values(["_distance", "name"]).iloc[0]
        return str(best["name"]), float(best["_distance"])

    eo_model, eo_distance = _pick_representative("EO-pretrained")
    imagenet_model, imagenet_distance = _pick_representative("ImageNet-pretrained")
    return ExemplarChoice(
        dataset=dataset,
        eo_model=eo_model,
        imagenet_model=imagenet_model,
        dataset_score=float(dataset_score),
        eo_distance=eo_distance,
        imagenet_distance=imagenet_distance,
    )
