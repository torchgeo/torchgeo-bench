"""Extrapolate $/inference and gCO2/inference from measured profile metrics.

Joins the ``method="profile"`` rows of an ``all_results.csv`` against the
GPU price and carbon-intensity tables in ``scripts/cost/``, then computes
per-cloud and per-region cost/emissions for an arbitrary inference
budget (defaults to 1M samples).

Assumes the throughput and energy in ``all_results.csv`` were measured
on the same GPU SKU that is being priced (passed via ``--measured-gpu``).
Extrapolating across GPU families is out of scope -- run the profile
sweep on the target GPU and re-join.

Usage:
    python scripts/analyze_compute_cost.py \\
        --results results/all_results.csv \\
        --measured-gpu "NVIDIA A100-SXM4-80GB" \\
        --samples 1000000 \\
        --top 20

    # All GPU types in the price table, only AWS, sorted by emissions
    python scripts/analyze_compute_cost.py --provider aws --sort kg_co2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

COST_DIR = Path(__file__).parent / "cost"
PROFILE_METRICS = (
    "throughput_samples_per_sec",
    "energy_wh_per_1k_samples",
    "gpu_power_w_avg",
    "params_m",
    "gmacs",
    "latency_ms_per_batch_p50",
    "peak_gpu_mem_gb",
)


def load_profile_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"] == "profile"]
    if df.empty:
        sys.exit(f"No method=profile rows in {path}. Run with eval.profile.enabled=true.")
    wide = df.pivot_table(
        index=["model", "name", "dataset", "bands"],
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def load_table(name: str, key: str) -> pd.DataFrame:
    with (COST_DIR / name).open() as f:
        return pd.DataFrame(yaml.safe_load(f)[key])


def compute_costs(
    profile: pd.DataFrame,
    prices: pd.DataFrame,
    carbon: pd.DataFrame,
    measured_gpu: str,
    n_samples: int,
) -> pd.DataFrame:
    prices = prices[prices["gpu_model"] == measured_gpu].copy()
    if prices.empty:
        sys.exit(
            f"No price entries for gpu_model={measured_gpu!r}. "
            f"Available: {sorted(set(prices['gpu_model']))}"
        )

    joined = profile.merge(prices, how="cross")
    joined = joined.merge(carbon, on="provider", how="left")

    # Throughput is per *single* GPU; instance price covers num_gpus, so
    # amortise to a per-GPU $/hr.
    joined["usd_per_hr_per_gpu"] = joined["usd_per_hr"] / joined["num_gpus"]

    seconds = n_samples / joined["throughput_samples_per_sec"]
    hours = seconds / 3600.0
    kwh = joined["energy_wh_per_1k_samples"] * (n_samples / 1000.0) / 1000.0

    joined["wall_seconds"] = seconds
    joined["kwh"] = kwh
    joined["usd"] = hours * joined["usd_per_hr_per_gpu"]
    joined["kg_co2"] = kwh * joined["gco2_per_kwh"] / 1000.0
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=Path("results/all_results.csv"))
    parser.add_argument(
        "--measured-gpu",
        default="NVIDIA A100-SXM4-80GB",
        help="GPU model string the profile rows were measured on (matches NVML name).",
    )
    parser.add_argument("--samples", type=int, default=1_000_000, help="Inferences to extrapolate.")
    parser.add_argument(
        "--provider", choices=("aws", "gcp", "azure"), help="Filter cloud provider."
    )
    parser.add_argument("--region", help="Filter region (exact match).")
    parser.add_argument("--top", type=int, default=20, help="Top-N rows to print.")
    parser.add_argument(
        "--sort",
        default="usd",
        choices=("usd", "kg_co2", "wall_seconds", "throughput_samples_per_sec"),
        help="Sort column (ascending).",
    )
    parser.add_argument("--out", type=Path, help="Write full joined table to CSV.")
    args = parser.parse_args()

    profile = load_profile_rows(args.results)
    prices = load_table("gpu_prices.yaml", "instances")
    carbon = load_table("carbon_intensity.yaml", "regions")

    df = compute_costs(profile, prices, carbon, args.measured_gpu, args.samples)
    if args.provider:
        df = df[df["provider"] == args.provider]
    if args.region:
        df = df[df["region"] == args.region]
    if df.empty:
        sys.exit("No rows match the requested filters.")

    df = df.sort_values(args.sort).reset_index(drop=True)

    show_cols = [
        "model",
        "dataset",
        "bands",
        "provider",
        "instance_type",
        "region",
        "throughput_samples_per_sec",
        "wall_seconds",
        "kwh",
        "usd",
        "kg_co2",
    ]
    print(df[show_cols].head(args.top).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {len(df)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
