#!/usr/bin/env python
"""Parallel benchmark sweep over model configs.

Automatically discovers all Hydra model YAML configs under `conf/model/` (including
subdirectories like `vit/`) and launches `torchgeo_bench.py` runs in parallel across
multiple GPUs.

Features:
    - Auto model list (no hardcoded ALL_TIMM_MODELS)
    - Round‑robin GPU assignment from global device list
    - Resume mode preserved (each subprocess uses Hydra resume logic)

Usage examples:
        # Run all configs across 2 GPUs with max 2 parallel jobs
        python run_timm_sweep.py --devices cuda:0 cuda:1 --max-parallel 2

        # Filter to specific models (names match YAML filename stem or subdir/stem)
        python run_timm_sweep.py --models resnet50 vit/vit_base_patch16_224 --devices cuda:0

        # Restrict datasets and change output
        python run_timm_sweep.py --datasets m-eurosat m-forestnet --output eurosat_forest.csv --devices cuda:0 cuda:1
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable
import time
import math
import threading
from queue import Queue, Empty


DEVICES = [0, 1, 2, 3, 4, 5, 6, 7]  # List of device IDs (integers) for GPU assignment


def discover_model_configs(root: Path) -> list[str]:
    """Discover Hydra model config identifiers under conf/model.

    For a YAML file at conf/model/resnet50.yaml -> 'resnet50'
    For nested path   conf/model/vit/vit_base_patch16_224.yaml -> 'vit/vit_base_patch16_224'
    """
    model_dir = root / "conf" / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model config directory not found: {model_dir}")
    models: list[str] = []
    for path in model_dir.rglob("*.yaml"):
        # Skip non-files just in case
        if not path.is_file():
            continue
        rel = path.relative_to(model_dir)
        # Convert to forward slash, drop .yaml
        stem = str(rel.as_posix()[:-5])  # remove '.yaml'
        models.append(stem)
    return sorted(models)


def build_command(model: str, output: str, datasets: list[str] | None, resume: bool, verbose: bool, device: int) -> list[str]:
    cmd = [
        "python",
        "torchgeo_bench.py",
        f"model={model}",
        f"output={output}",
        f"device={device}",
        f"verbose={verbose}",
        f"resume={resume}",
        #"eval.skip_linear=True",  # keep sweep fast
    ]
    if datasets:
        dataset_str = "[" + ",".join(datasets) + "]"
        cmd.append(f"dataset.names={dataset_str}")
    return cmd


def run_job(cmd: list[str]) -> int:
    print(f"\n{'='*60}\nLaunching: {' '.join(cmd)}\n{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Parallel benchmark sweep over discovered model configs"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Explicit list of model identifiers (default: discover all). Use nested names like vit/vit_base_patch16_224.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="List of dataset names (default: all datasets)",
    )
    parser.add_argument(
        "--output",
        default="timm_sweep_results.csv",
        help="Output CSV file (default: timm_sweep_results.csv)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode (recompute everything)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    # Removed --devices / --max-parallel / --poll-interval; use DEVICES global.
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent
    discovered = discover_model_configs(repo_root)
    if args.models:
        requested = args.models
        # validate against discovered
        missing = [m for m in requested if m not in discovered]
        if missing:
            print(f"Warning: {len(missing)} requested models not found in configs: {missing}")
        models = [m for m in requested if m in discovered] + [m for m in requested if m not in discovered]
    else:
        models = discovered

    devices = DEVICES
    
    print("Benchmark Sweep Configuration:")
    sample_models = ", ".join(models[:3]) + (f", ... and {len(models)-3} more" if len(models) > 3 else "")
    print(f"  Models: {len(models)} ({sample_models})")
    print(f"  Datasets: {'all' if not args.datasets else ', '.join(args.datasets)}")
    print(f"  Output: {args.output}")
    print(f"  Resume: {not args.no_resume}")
    print(f"  Devices: {devices} (threads={len(devices)})")
    print()
    
    # Threaded dispatch: one worker thread per device
    failed: list[str] = []
    durations: dict[str, float] = {}
    results: dict[str, int] = {}
    start_times: dict[str, float] = {}

    job_queue: Queue[tuple[str, int]] = Queue()
    for idx, model in enumerate(models):
        # device assignment will be done by workers; store desired device index mapping
        device = devices[idx % len(devices)]
        job_queue.put((model, device))

    # Limit number of workers to max_parallel or available devices
    worker_devices = devices

    def worker(device: str) -> None:
        while True:
            try:
                model, dev_assigned = job_queue.get(timeout=1)
            except Empty:
                break
            # Ensure device chosen matches assigned round-robin mapping
            dev = dev_assigned
            cmd = build_command(
                model=model,
                output=args.output,
                datasets=args.datasets,
                resume=not args.no_resume,
                verbose=args.verbose,
                device=dev,
            )
            start_times[model] = time.time()
            print(f"[START] {model} on {dev}")
            rc = run_job(cmd)
            dur = time.time() - start_times[model]
            durations[model] = dur
            results[model] = rc
            if rc != 0:
                failed.append(model)
                print(f"[FAILED] {model} rc={rc} time={dur/60:.2f}m")
            else:
                print(f"[DONE]  {model} time={dur/60:.2f}m")
            job_queue.task_done()

    threads: list[threading.Thread] = []
    for d in worker_devices:
        t = threading.Thread(target=worker, args=(d,), daemon=True)
        t.start()
        threads.append(t)

    # Wait for completion
    for t in threads:
        t.join()
    
    # Summary
    print(f"\n\n{'='*60}")
    print("Sweep Summary")
    print(f"{'='*60}")
    print(f"Total models: {len(models)}")
    print(f"Successful: {len(models) - len(failed)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed models: {', '.join(failed)}")
    # Rough throughput estimate
    if models:
        successful = [m for m in models if m not in failed]
        avg_time = sum(durations[m] for m in successful) / max(1, len(successful))
        print(f"Avg time per successful model: {avg_time/60:.2f} min")
        fastest = sorted(successful, key=lambda m: durations[m])[:3]
        slowest = sorted(successful, key=lambda m: durations[m], reverse=True)[:3]
        if successful:
            print(f"Fastest: {', '.join(f'{m} ({durations[m]/60:.1f}m)' for m in fastest)}")
            print(f"Slowest: {', '.join(f'{m} ({durations[m]/60:.1f}m)' for m in slowest)}")
    
    if failed:
        sys.exit(1)
    print(f"Results appended to: {args.output}")
    sys.exit(0)


if __name__ == "__main__":
    main()
