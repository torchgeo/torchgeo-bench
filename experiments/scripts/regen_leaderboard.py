"""Regenerate ``docs/_static/ranking_explorer.html`` from ``all_results.csv``.

Builds a TabArena-style, methodology-aware leaderboard of frozen geospatial
foundation models. Every ranking is precomputed offline with ``evaluma`` and
inlined into the page as JSON: the leaderboard reorders both with the chosen
aggregation (average rank / ELO / improvability) and with the condition slice
(task / probe / bands / pooling / normalization). A per-axis Kendall tau-b
rank-sensitivity readout quantifies how much the order moves when a single
condition flips.

Usage::

    python experiments/scripts/regen_leaderboard.py [--csv results/all_results.csv]
"""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "all_results.csv"
HTML_PATH = ROOT / "docs" / "_static" / "ranking_explorer.html"

# Rows whose ``name`` contains any of these are single-dataset / experimental
# OlmoEarth ablations, not benchmark entries.
ABLATION_TAGS = ("_lsat_", "_naip_", "_sar_")

# Raw metric name -> canonical name registered in the evaluma metric registry.
METRIC_MAP = {"accuracy": "accuracy", "micro_mAP": "map"}

# Fixed per-metric normalization bounds so evaluma normalization is
# roster-independent (otherwise it uses observed min/max and rescales on every
# roster change, emitting a UserWarning).
METRIC_BOUNDS = {"accuracy": (0.0, 1.0), "map": (0.0, 1.0)}
DEFAULT_RANDOM_SEED = 0
NORMALIZATION_ORDER = ["bandspec_zscore", "model_native"]

# base ``name`` slug -> (display name, family, is_baseline)
MODEL_META: dict[str, tuple[str, str, bool]] = {
    # baselines
    "rcf": ("Random conv. features", "Baseline", True),
    "imagestats": ("Image statistics", "Baseline", True),
    # OlmoEarth
    "olmoearth_v1_nano": ("OlmoEarth v1 nano", "OlmoEarth", False),
    "olmoearth_v1_tiny": ("OlmoEarth v1 tiny", "OlmoEarth", False),
    "olmoearth_v1_base": ("OlmoEarth v1 base", "OlmoEarth", False),
    "olmoearth_v1_large": ("OlmoEarth v1 large", "OlmoEarth", False),
    "olmoearth_v1_1_nano": ("OlmoEarth v1.1 nano", "OlmoEarth", False),
    "olmoearth_v1_1_tiny": ("OlmoEarth v1.1 tiny", "OlmoEarth", False),
    "olmoearth_v1_1_base": ("OlmoEarth v1.1 base", "OlmoEarth", False),
    # CROMA
    "tgeo_croma_base": ("CROMA base", "CROMA", False),
    "tgeo_croma_large": ("CROMA large", "CROMA", False),
    # DOFA
    "tgeo_dofa_base": ("DOFA base", "DOFA", False),
    "tgeo_dofa_large": ("DOFA large", "DOFA", False),
    # misc torchgeo backbones
    "tgeo_earthloc_s2_resnet50": ("EarthLoc ResNet-50", "EarthLoc", False),
    "tgeo_panopticon": ("Panopticon", "Panopticon", False),
    "tgeo_resnet18_s2rgb_seco": ("SeCo ResNet-18", "SeCo", False),
    "tgeo_resnet50_s2rgb_seco": ("SeCo ResNet-50", "SeCo", False),
    "tgeo_resnet50_fmow_gassl": ("GASSL ResNet-50", "GASSL", False),
    "tgeo_resnet50_s2all_moco": ("MoCo ResNet-50 (MSI)", "MoCo", False),
    "tgeo_resnet50_s2rgb_moco": ("MoCo ResNet-50 (RGB)", "MoCo", False),
    "tgeo_scalemae_large_fmow": ("ScaleMAE large", "ScaleMAE", False),
    "tgeo_swinv2b_s2rgb_satlas_mi": ("SatlasNet Swin-v2-B (MI)", "SatlasNet", False),
    "tgeo_swinv2b_s2rgb_satlas_si": ("SatlasNet Swin-v2-B (SI)", "SatlasNet", False),
    "tgeo_swinv2t_s2rgb_satlas_mi": ("SatlasNet Swin-v2-T (MI)", "SatlasNet", False),
    "tgeo_swinv2t_s2rgb_satlas_si": ("SatlasNet Swin-v2-T (SI)", "SatlasNet", False),
    # Clay
    "tt_clay_v1_5_base": ("Clay v1.5 base", "Clay", False),
    # Prithvi
    "tt_prithvi_eo_v1_100": ("Prithvi-EO-v1 100M", "Prithvi", False),
    "tt_prithvi_eo_v2_100_tl": ("Prithvi-EO-v2 100M (TL)", "Prithvi", False),
    "tt_prithvi_eo_v2_300": ("Prithvi-EO-v2 300M", "Prithvi", False),
    "tt_prithvi_eo_v2_300_tl": ("Prithvi-EO-v2 300M (TL)", "Prithvi", False),
    "tt_prithvi_eo_v2_600": ("Prithvi-EO-v2 600M", "Prithvi", False),
    # TerraMind
    "tt_terramind_v1_base": ("TerraMind v1 base", "TerraMind", False),
    "tt_terramind_v1_large": ("TerraMind v1 large", "TerraMind", False),
    # DINOv3
    "vit_large_patch16_dinov3": ("DINOv3 ViT-L/16", "DINOv3", False),
    "vit_large_patch16_dinov3sat": ("DINOv3-SAT ViT-L/16", "DINOv3", False),
}


def resolve_meta(slug: str) -> tuple[str, str, bool]:
    """Resolve a base-model slug to ``(display, family, is_baseline)``.

    Unmapped slugs fall back to ``(slug, slug, False)``.
    """
    return MODEL_META.get(slug, (slug, slug, False))


def ordered_values(values: pd.Series, preferred: list[str]) -> list[str]:
    """Return observed string values with a preferred stable order first."""
    observed = [str(v) for v in values.dropna().unique()]
    pinned = [value for value in preferred if value in observed]
    trailing = sorted(value for value in observed if value not in set(preferred))
    return pinned + trailing


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw ``all_results.csv`` rows into a canonical long-format frame.

    Drops OlmoEarth ablation rows and non-quality metrics; derives ``pooling``
    (``cls-token`` for ``_cls``-suffixed names, else ``patch-mean``) and a base
    model name; derives the semantic ``bandclass`` (``RGB`` if ``bands == "rgb"``
    else ``Multispectral``); canonicalizes the metric name; and keeps
    ``normalization`` explicit as a first-class condition axis. Does not
    collapse duplicate score cells: ambiguous source rows remain present so the
    slice builder can reject seated duplicates loudly.

    Returns:
        DataFrame with columns ``base_model, dataset, task, probe, bandclass,
        pooling, normalization, metric, score`` — one row per evaluated result.
    """
    df = df.copy()
    name = df["name"].astype(str)

    # drop single-dataset ablation rows
    ablation = pd.Series(False, index=df.index)
    for tag in ABLATION_TAGS:
        ablation |= name.str.contains(tag, regex=False)
    df = df[~ablation]
    name = name[~ablation]

    # keep only quality-metric rows and canonicalize the metric name
    df = df[df["metric_name"].isin(METRIC_MAP)].copy()
    df["metric"] = df["metric_name"].map(METRIC_MAP)

    # pooling + base model from the ``_cls`` suffix
    name = df["name"].astype(str)
    is_cls = name.str.endswith("_cls")
    df["pooling"] = is_cls.map({True: "cls-token", False: "patch-mean"})
    df["base_model"] = name.where(~is_cls, name.str.slice(0, -4))

    # semantic band class
    df["bandclass"] = (df["bands"].astype(str) == "rgb").map({True: "RGB", False: "Multispectral"})

    df["probe"] = df["method"]
    df["task"] = "classification"
    df["normalization"] = df["normalization"].astype("string").str.strip()
    missing_norm = df["normalization"].isna() | (df["normalization"] == "")
    if missing_norm.any():
        preview = df.loc[missing_norm, ["name", "dataset", "method", "bands", "metric_name"]].head(
            10
        )
        raise ValueError(
            "Ranking explorer quality rows must record `normalization`. "
            f"Found {int(missing_norm.sum())} missing rows.\n"
            f"{preview.to_string(index=False)}"
        )
    df["normalization"] = df["normalization"].astype(str)
    df["score"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df = df.dropna(subset=["score"])

    keys = [
        "base_model",
        "dataset",
        "task",
        "probe",
        "bandclass",
        "pooling",
        "normalization",
        "metric",
    ]
    return df[keys + ["score"]].reset_index(drop=True)


def extract_params(raw: pd.DataFrame) -> dict[str, float]:
    """Map each base model to its measured frozen-backbone size in millions.

    Reads the recorded ``params_m`` metric (``sum(p.numel())/1e6`` of the frozen
    backbone), drops ablation rows, strips the ``_cls`` readout suffix, and
    collapses to one value per base model (max across the model's rows;
    cross-band spread is <=1.4%, from the input-projection layer). Baselines are
    recorded as ``0.0``.

    Returns:
        ``{base_model: params_m}`` for every model with a recorded value.
    """
    df = raw[raw["metric_name"] == "params_m"].copy()
    name = df["name"].astype(str)
    ablation = pd.Series(False, index=df.index)
    for tag in ABLATION_TAGS:
        ablation |= name.str.contains(tag, regex=False)
    df = df[~ablation]
    name = df["name"].astype(str)
    df["base_model"] = name.where(~name.str.endswith("_cls"), name.str.slice(0, -4))
    df["params_m"] = pd.to_numeric(df["metric_value"], errors="coerce")
    vals = df.dropna(subset=["params_m"]).groupby("base_model")["params_m"].max()
    return {model: float(v) for model, v in vals.items()}


def _describe_slice_key(
    task: str,
    probe: str,
    bandclass: str,
    pooling: str,
    normalization: str,
) -> str:
    return (
        f"task={task}, probe={probe}, bandclass={bandclass}, pooling={pooling}, "
        f"normalization={normalization}"
    )


def _raise_on_duplicate_seated_cells(
    sub: pd.DataFrame,
    complete: pd.Index,
    *,
    task: str,
    probe: str,
    bandclass: str,
    pooling: str,
    normalization: str,
) -> None:
    """Reject ambiguous seated rows instead of silently collapsing them."""
    seated = sub[sub["base_model"].isin(complete)].copy()
    dup_keys = ["base_model", "dataset", "metric"]
    dup_mask = seated.duplicated(dup_keys, keep=False)
    if not dup_mask.any():
        return

    offenders = seated.loc[dup_mask, dup_keys + ["score"]].sort_values(dup_keys + ["score"])
    raise ValueError(
        "Duplicate seated leaderboard cells remain after preserving `normalization` "
        f"for slice {_describe_slice_key(task, probe, bandclass, pooling, normalization)}.\n"
        "Each seated (model, dataset, metric) cell must come from exactly one "
        "evaluation row.\n"
        f"{offenders.to_string(index=False)}"
    )


def build_slice(
    harmonized: pd.DataFrame,
    task: str,
    probe: str,
    bandclass: str,
    pooling: str,
    normalization: str,
):
    """Build the evaluma ``Benchmark`` for one condition slice (clean path).

    Restricts ``harmonized`` to the
    ``(task, probe, bandclass, pooling, normalization)`` slice, keeps only
    models with complete dataset coverage, rejects residual duplicate seated
    cells, and loads a dense model x dataset benchmark via
    ``evaluma.load_df(..., drop_incomplete=True)`` with fixed, roster-independent
    normalization bounds.

    Returns:
        ``(bench, excluded)`` where ``bench`` is the ``Benchmark`` over
        fully-complete models (or ``None`` when the slice has no complete
        roster) and ``excluded`` is a list of
        ``{"model", "n_tasks"}`` for partial-coverage models (sorted by model).
    """
    import evaluma

    sub = harmonized[
        (harmonized["task"] == task)
        & (harmonized["probe"] == probe)
        & (harmonized["bandclass"] == bandclass)
        & (harmonized["pooling"] == pooling)
        & (harmonized["normalization"] == normalization)
    ]
    datasets = sorted(sub["dataset"].unique())
    total = len(datasets)
    if total == 0:
        return None, []

    coverage = sub.groupby("base_model")["dataset"].nunique()

    complete = coverage[coverage == total].index
    excluded = [{"model": model, "n_tasks": int(n)} for model, n in coverage.items() if n < total]
    excluded.sort(key=lambda e: e["model"])
    if len(complete) == 0:
        return None, excluded

    _raise_on_duplicate_seated_cells(
        sub,
        complete,
        task=task,
        probe=probe,
        bandclass=bandclass,
        pooling=pooling,
        normalization=normalization,
    )

    frame = (
        sub[sub["base_model"].isin(complete)]
        .rename(columns={"base_model": "model"})[["model", "dataset", "metric", "score"]]
        .reset_index(drop=True)
    )
    bench = evaluma.load_df(frame, metric_type_bounds=METRIC_BOUNDS, drop_incomplete=True)
    return bench, excluded


def avg_rank(bench) -> list[dict]:
    """TabArena-style average rank from the slice's raw score matrix.

    Ranks each dataset column with rank 1 = best (highest score) and averages
    per model. Independent of evaluma normalization. Lower is better.

    Returns:
        Rows ``{"model", "overall"}`` sorted ascending (rank 1 = best).
    """
    scores = bench.scores_
    mean_ranks = scores.rank(ascending=False, axis=0).mean(axis=1).sort_values()
    return [{"model": model, "overall": float(rank)} for model, rank in mean_ranks.items()]


def elo(
    bench, n_bootstrap: int = 1000, random_state: int | None = DEFAULT_RANDOM_SEED
) -> list[dict]:
    """MLE ELO ratings with battle-within-task bootstrap CIs. Higher is better.

    Returns:
        Rows ``{"model", "overall", "overall_ci": (low, high)}`` sorted by ELO
        descending (rank 1 = best).
    """
    table = bench.elo_ranking(n_bootstrap=n_bootstrap, random_state=random_state).table.sort_values(
        "ELO", ascending=False
    )
    return [
        {
            "model": row.model,
            "overall": float(row.ELO),
            "overall_ci": (float(row.CI_low), float(row.CI_high)),
        }
        for row in table.itertuples(index=False)
    ]


def improvability(bench) -> list[dict]:
    """Mean percent error-reduction to the per-dataset best. Lower is better.

    Relies on the canonical ``accuracy``/``map`` metric names registered in the
    evaluma metric registry (error optimum 1.0). The per-dataset best scores 0.

    Returns:
        Rows ``{"model", "overall"}`` sorted ascending (0 = best).
    """
    table = bench.improvability_ranking().table.sort_values("improvability")
    return [
        {"model": row.model, "overall": float(row.improvability)}
        for row in table.itertuples(index=False)
    ]


# condition axis -> (slice-key column, value A, value B)
BASE_SENSITIVITY_AXES = {
    "probe": ("probe", "linear", "knn5"),
    "bands": ("bandclass", "RGB", "Multispectral"),
    "pooling": ("pooling", "patch-mean", "cls-token"),
}

# Flag a pairwise sensitivity as small-N when the shared roster is below this;
# the cls slices legitimately collapse to 6-7 models, so surface (not suppress).
SMALL_N_MODELS = 8


def sensitivity_axes(normalizations: list[str]) -> dict[str, tuple[str, str, str]]:
    """Return the pairwise condition flips surfaced in the sensitivity UI."""
    axes = dict(BASE_SENSITIVITY_AXES)
    if len(normalizations) >= 2:
        axes["normalization"] = ("normalization", normalizations[0], normalizations[1])
    return axes


def axis_sensitivity(
    harmonized: pd.DataFrame,
    slice_key: dict,
    axis: str,
    aggregation: str,
    axes: dict[str, tuple[str, str, str]],
) -> dict:
    """Pairwise Kendall tau-b rank sensitivity for flipping one condition axis.

    Builds both conditions' clean-path benchmarks, restricts them to the
    intersection of their complete models and datasets (``rank_sensitivity``
    requires identical sets), rebuilds on the intersection, and computes tau
    under the *selected* ``aggregation`` — so tau moves across aggregations
    exactly when they disagree on within-slice model ordering. Point estimate
    only (no CI; the panel never displays one).

    Returns:
        ``{cond_a, cond_b, tau, n_models, n_datasets, small_n}``; ``tau`` is
        ``None`` when undefined (e.g. a degenerate roster).
    """
    col, val_a, val_b = axes[axis]
    bench_a, _ = build_slice(harmonized, **{**slice_key, col: val_a})
    bench_b, _ = build_slice(harmonized, **{**slice_key, col: val_b})
    if bench_a is None or bench_b is None:
        return {
            "cond_a": val_a,
            "cond_b": val_b,
            "tau": None,
            "n_models": 0,
            "n_datasets": 0,
            "small_n": True,
        }

    models = sorted(set(bench_a.models_) & set(bench_b.models_))
    datasets = sorted(set(bench_a.datasets_) & set(bench_b.datasets_))
    if not models or not datasets:
        return {
            "cond_a": val_a,
            "cond_b": val_b,
            "tau": None,
            "n_models": 0,
            "n_datasets": 0,
            "small_n": True,
        }
    sub_a = bench_a.select_models(models).select_datasets(datasets)
    sub_b = bench_b.select_models(models).select_datasets(datasets)

    # n_bootstrap=0: the non-aggregate rankers are point-estimate only (no CI),
    # and passing 0 suppresses evaluma's "bootstrap ignored" warning.
    res = sub_a.rank_sensitivity(sub_b, val_a, val_b, ranker=aggregation, n_bootstrap=0)
    tau = res.tau
    return {
        "cond_a": val_a,
        "cond_b": val_b,
        "tau": None if tau is None or math.isnan(tau) else float(tau),
        "n_models": len(models),
        "n_datasets": len(datasets),
        "small_n": len(models) < SMALL_N_MODELS,
    }


# Condition-axis grid enumerated into the ranking cube.
TASKS = ["classification"]
PROBES = ["linear", "knn5"]
BANDCLASSES = ["RGB", "Multispectral"]
POOLINGS = ["patch-mean", "cls-token"]

# aggregation name -> (function, overall-column metadata)
AGG_META = {
    "avg_rank": {"label": "Average rank", "direction": "lower-is-better", "unit": "rank"},
    "elo": {"label": "ELO", "direction": "higher-is-better", "unit": "elo"},
    "improvability": {"label": "Improvability", "direction": "lower-is-better", "unit": "%"},
}

DEFAULT_SLICE = {
    "task": "classification",
    "probe": "linear",
    "bandclass": "RGB",
    "pooling": "patch-mean",
    "normalization": "bandspec_zscore",
    "aggregation": "avg_rank",
}


def _enrich_rows(bench, ranked: list[dict]) -> list[dict]:
    """Attach display metadata and per-task raw scores to ranked rows."""
    scores = bench.scores_
    n_tasks = len(bench.datasets_)
    rows = []
    for r in ranked:
        model = r["model"]
        display, family, is_baseline = resolve_meta(model)
        ci = r.get("overall_ci")
        rows.append(
            {
                "model": model,
                "display": display,
                "family": family,
                "is_baseline": is_baseline,
                "overall": r["overall"],
                "overall_ci": list(ci) if ci is not None else None,
                "n_tasks": n_tasks,
                "per_task": {ds: float(v) for ds, v in scores.loc[model].items()},
            }
        )
    return rows


def assemble(
    harmonized: pd.DataFrame,
    n_bootstrap: int = 1000,
    params: dict | None = None,
    random_state: int | None = DEFAULT_RANDOM_SEED,
):
    """Assemble the full ranking cube and rank-sensitivity readouts.

    Args:
        harmonized: Canonical long frame from :func:`harmonize`.
        n_bootstrap: Bootstrap resamples for ELO CIs.
        params: Optional ``{base_model: params_m}`` map from
            :func:`extract_params`; emitted into ``MODEL_META`` as ``params_m``.

    Returns:
        ``(RANKINGS, SENSITIVITY, MODEL_META_OUT, DEFAULT_SLICE)`` where

        * ``RANKINGS[task][probe][bandclass][pooling][normalization][aggregation]`` carries
          ``{rows, overall_meta, excluded}``;
        * ``SENSITIVITY[task][probe][bandclass][pooling][normalization][aggregation][axis]``
          carries the pairwise Kendall tau-b dict, recomputed under each
          aggregation's ordering;
        * ``MODEL_META_OUT`` maps every seated slug to ``{display, family,
          is_baseline, params_m}`` (``params_m`` is ``None`` when unrecorded).
    """
    params = params or {}
    normalizations = ordered_values(harmonized["normalization"], NORMALIZATION_ORDER)
    axes = sensitivity_axes(normalizations)
    default_slice = dict(DEFAULT_SLICE)
    if normalizations and default_slice["normalization"] not in normalizations:
        default_slice["normalization"] = normalizations[0]
    rankings: dict = {}
    sensitivity: dict = {}
    seated: set[str] = set()

    for task in TASKS:
        for probe in PROBES:
            for bandclass in BANDCLASSES:
                for pooling in POOLINGS:
                    for normalization in normalizations:
                        key = {
                            "task": task,
                            "probe": probe,
                            "bandclass": bandclass,
                            "pooling": pooling,
                            "normalization": normalization,
                        }
                        bench, excluded = build_slice(harmonized, **key)
                        if bench is None or len(bench.models_) < 2:
                            continue  # nothing to rank
                        seated.update(bench.models_)

                        ranked_by_agg = {
                            "avg_rank": avg_rank(bench),
                            "elo": elo(bench, n_bootstrap=n_bootstrap, random_state=random_state),
                            "improvability": improvability(bench),
                        }
                        agg_block = {
                            name: {
                                "rows": _enrich_rows(bench, ranked),
                                "overall_meta": AGG_META[name],
                                "excluded": excluded,
                            }
                            for name, ranked in ranked_by_agg.items()
                        }
                        (
                            rankings.setdefault(task, {})
                            .setdefault(probe, {})
                            .setdefault(bandclass, {})
                            .setdefault(pooling, {})
                        )[normalization] = agg_block

                        agg_sensitivity = {
                            aggregation: {
                                axis: axis_sensitivity(harmonized, key, axis, aggregation, axes)
                                for axis in axes
                            }
                            for aggregation in AGG_META
                        }
                        (
                            sensitivity.setdefault(task, {})
                            .setdefault(probe, {})
                            .setdefault(bandclass, {})
                            .setdefault(pooling, {})
                        )[normalization] = agg_sensitivity

    model_meta_out = {}
    for slug in sorted(seated):
        display, family, is_baseline = resolve_meta(slug)
        pm = params.get(slug)
        model_meta_out[slug] = {
            "display": display,
            "family": family,
            "is_baseline": is_baseline,
            "params_m": None if pm is None else float(pm),
        }

    return rankings, sensitivity, model_meta_out, default_slice


def inline_into_html(html_text: str, blocks: dict) -> str:
    """Replace each ``const NAME = …;`` block in ``html_text`` with JSON.

    ``blocks`` maps a constant name (``RANKINGS``/``SENSITIVITY``/``MODEL_META``/
    ``DEFAULT_SLICE``) to a JSON-serializable object. Relies on the emitted JSON
    containing no ``;`` outside strings (true for this data), so a non-greedy
    match to the first ``;`` terminates each declaration.
    """
    text = html_text
    for name, obj in blocks.items():
        payload = json.dumps(obj, separators=(",", ":"), allow_nan=False)
        pattern = re.compile(rf"const {name} = .*?;", re.DOTALL)
        if not pattern.search(text):
            raise SystemExit(f"Could not locate `const {name} = …;` in HTML.")
        text = pattern.sub(lambda _m, n=name, p=payload: f"const {n} = {p};", text, count=1)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH), help="Path to all_results.csv.")
    parser.add_argument("--html", default=str(HTML_PATH), help="Path to the leaderboard HTML.")
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Bootstrap resamples for ELO / sensitivity CIs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible ELO bootstrap confidence intervals.",
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.csv, low_memory=False)
    harmonized = harmonize(raw)
    params = extract_params(raw)
    rankings, sensitivity, model_meta, default_slice = assemble(
        harmonized, n_bootstrap=args.bootstrap, params=params, random_state=args.seed
    )
    blocks = {
        "RANKINGS": rankings,
        "SENSITIVITY": sensitivity,
        "MODEL_META": model_meta,
        "DEFAULT_SLICE": default_slice,
    }

    html_path = Path(args.html)
    text = inline_into_html(html_path.read_text(), blocks)
    html_path.write_text(text)

    n_slices = sum(
        len(norms)
        for task in rankings.values()
        for probe in task.values()
        for bandclass in probe.values()
        for norms in bandclass.values()
    )
    print(
        f"Wrote {html_path}: {n_slices} condition slices, {len(model_meta)} models, "
        f"default {default_slice['task']}·{default_slice['probe']}·"
        f"{default_slice['bandclass']}·{default_slice['pooling']}·"
        f"{default_slice['normalization']}"
    )


if __name__ == "__main__":
    main()
