"""Shared CLI, multi-GPU scheduling and atomic CSV reporting for decoder studies."""

import argparse
import csv
import json
import logging
import queue
import signal
import statistics
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import _segmentation_convergence as convergence
import _segmentation_features as features
import torch
from _seg_sweep_common import BaseGpuRunner, RunnerConfig, parse_gpus, resolve_path
from filelock import FileLock

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
LAYER_PRESETS = {
    "deep_single": ["blocks.11"],
    "late_single": ["blocks.8"],
    "mid_single": ["blocks.5"],
    "shallow_single": ["blocks.2"],
    "late_four": ["blocks.11", "blocks.10", "blocks.9", "blocks.8"],
    "spread_four": ["blocks.11", "blocks.8", "blocks.5", "blocks.2"],
    "early_four": ["blocks.5", "blocks.4", "blocks.3", "blocks.2"],
}


@dataclass(frozen=True)
class Job:
    """One decoder, connection set, optimizer, LR and paired seed."""

    trial: convergence.TrialConfig
    layer_group: str
    layers: tuple[str, ...]

    @property
    def job_id(self) -> str:
        """Return a scheduling-independent identifier, without redundant group aliases."""
        return features.identity({"trial": asdict(self.trial), "layers": self.layers})


@dataclass(frozen=True)
class StudyConfig(RunnerConfig):
    """Runtime locations and resources, deliberately outside scientific identity."""

    script: Path
    cache: Path
    output_dir: Path
    resume: bool


def selected_gpus(value: str, *, dry_run: bool = False) -> list[int]:
    """Select all visible GPUs by default; reject repeats rather than oversubscribe."""
    if dry_run and not torch.cuda.device_count():
        gpus = [0] if value == "all" else [int(item) for item in value.split(",")]
        logger.info("No CUDA devices visible: dry-run commands use hypothetical GPU indices.")
        if not gpus or min(gpus) < 0:
            raise ValueError("GPU indices must be nonnegative.")
    else:
        gpus = parse_gpus(value)
    if len(set(gpus)) != len(gpus):
        raise ValueError("Duplicate GPU indices are not allowed.")
    return gpus


def layer_groups(args: argparse.Namespace, suite: str) -> dict[str, list[str]]:
    """Validate named ordered groups and refuse aliases for identical connections."""
    if suite == "optimizer":
        features.validate_layers(args.layers)
        return {"connections": list(args.layers)}
    if args.layer_groups_json:
        groups = json.loads(args.layer_groups_json, object_pairs_hook=unique_group_names)
        if not isinstance(groups, dict) or not groups:
            raise ValueError("--layer-groups-json must be a nonempty JSON object.")
    else:
        if len(set(args.layer_groups)) != len(args.layer_groups):
            raise ValueError("Duplicate layer group names.")
        groups = {name: LAYER_PRESETS[name] for name in args.layer_groups}
    for name, layers in groups.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(layers, list):
            raise ValueError("Layer groups map nonempty names to lists of layer names.")
        features.validate_layers(layers)
    if len({tuple(layers) for layers in groups.values()}) != len(groups):
        raise ValueError("Duplicate layer connections under different group names.")
    return groups


def unique_group_names(pairs: list[tuple[str, object]]) -> dict:
    """Reject duplicate JSON keys instead of silently replacing a requested group."""
    groups = dict(pairs)
    if len(groups) != len(pairs):
        raise ValueError("Duplicate layer group names in JSON.")
    return groups


def build_jobs(args: argparse.Namespace, suite: str) -> list[Job]:
    """Build paired seeds with all Adam jobs before L-BFGS and explicit compatibility."""
    groups = layer_groups(args, suite)
    for values in (args.heads, args.seeds, args.adam_lrs, args.lbfgs_lrs):
        if not values or len(set(values)) != len(values):
            raise ValueError("Empty or duplicate heads, seeds or learning rates.")
    options = {
        name: getattr(args, name)
        for name in convergence.TrialConfig.__dataclass_fields__
        if name not in ("head", "optimizer", "lr", "seed")
    }
    jobs = []
    for optimizer, rates in (("adam", args.adam_lrs), ("lbfgs", args.lbfgs_lrs)):
        if suite == "layer" and optimizer != "adam":
            continue
        if args.phase not in ("both", optimizer):
            continue
        for head in args.heads:
            for group, layers in groups.items():
                selected = layers
                if suite == "optimizer" and head == "patch_linear":
                    selected = layers[:1]
                    logger.info("patch_linear explicitly uses only deepest layer %s.", selected)
                features.check_head_layers(head, selected)
                jobs.extend(
                    Job(
                        convergence.TrialConfig(head, optimizer, lr, seed=seed, **options),
                        group,
                        tuple(selected),
                    )
                    for seed in args.seeds
                    for lr in rates
                )
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Empty or ambiguous duplicate trial grid.")
    return jobs


def layer_union(jobs: list[Job]) -> list[str]:
    """Preserve first occurrence order while extracting every required connection once."""
    return list(dict.fromkeys(name for job in jobs for name in job.layers))


def job_identity(job: Job, metadata: dict) -> dict:
    """Bind a trial to its named subset and the validated union cache."""
    indices = [metadata["spec"]["layers"].index(name) for name in job.layers]
    selected = {
        **metadata,
        "selected_layers": list(job.layers),
        "selected_feature_shapes": [metadata["feature_shapes"][i] for i in indices],
    }
    return convergence.scientific_identity(job.trial, selected)


def result_row(job: Job, metadata: dict, result: dict | None) -> dict:
    """Flatten scientific status, selected performance and comparable timing counters."""
    spec = metadata["spec"]
    selected = (result or {}).get("selected") or {}
    final = (result or {}).get("final") or {}
    timing = (result or {}).get("timings_seconds") or {}
    ci = (result or {}).get("test_miou_ci95", [None, None])
    row = {
        "job_id": job.job_id,
        "trial_key": features.identity(job_identity(job, metadata)),
        "dataset": spec["dataset"],
        "model": spec["model"],
        "head": job.trial.head,
        "layer_group": job.layer_group,
        "layers": json.dumps(job.layers),
        "layer_count": len(job.layers),
        "feature_shapes": json.dumps(job_identity(job, metadata)["selected_feature_shapes"]),
        "bands": spec["bands"],
        "image_size": spec["image_size"],
        "normalization": spec["normalization"],
        "optimizer": job.trial.optimizer,
        "lr": job.trial.lr,
        "seed": job.trial.seed,
        "batch_size": job.trial.batch_size,
        "hidden_dim": job.trial.hidden_dim,
        "status": (result or {}).get("status", "incomplete"),
        "stop_reason": (result or {}).get("stop_reason"),
        "stationary": (result or {}).get("stationary"),
        "failure": (result or {}).get("failure"),
        "best_val_miou": selected.get("val_miou"),
        "best_val_ce": selected.get("val_ce"),
        "test_miou": (result or {}).get("test", {}).get("miou"),
        "test_miou_ci95_low": ci[0],
        "test_miou_ci95_high": ci[1],
        "parameter_count": (result or {}).get("parameter_count"),
        "trainable_parameter_count": (result or {}).get("trainable_parameter_count"),
        "optimization_seconds": timing.get("optimization"),
        "training_wall_seconds": (result or {}).get("training_wall_seconds"),
        "selected_optimization_seconds": selected.get("optimization_seconds"),
        "selected_training_wall_seconds": selected.get("training_wall_seconds"),
        "selected_iterations": selected.get("iterations"),
        "iterations": (result or {}).get("iterations"),
        "final_train_ce": final.get("train_ce"),
        "final_gradient_inf_norm": final.get("gradient_inf_norm"),
        "initial_state_sha256": (result or {}).get("initial_state_sha256"),
        "cache_key": metadata["key"],
        "cache_tensor_sha256": metadata["tensor_sha256"],
        "hardware_history": json.dumps((result or {}).get("hardware_history", []), sort_keys=True),
        "mixed_hardware_timings": (result or {}).get("mixed_hardware_timings"),
    }
    for name in (
        "blocks",
        "epochs",
        "optimizer_step_calls",
        "optimizer_updates",
        "samples",
        "calibration_passes",
        "unchanged_blocks",
        "stalled_iteration_blocks",
    ):
        row[name] = (result or {}).get("counters", {}).get(name)
    for name in ("closure_evaluations", "monitor_evaluations"):
        row[name] = (result or {}).get(name)
    for name in (
        "lbfgs_n_iter",
        "lbfgs_func_evals",
        "lbfgs_history_length",
        "optimization_data_passes",
        "recent_ce_range",
        "recent_ce_half_mean_decrease",
        "recent_ce_tolerance",
    ):
        row[name] = final.get(name)
    row.update(
        {
            f"{stage}_seconds": timing.get(stage)
            for stage in convergence.STAGES
            if stage != "optimization"
        }
    )
    return row


def validation_summary(rows: list[dict]) -> list[dict]:
    """Choose one LR by mean validation across paired seeds, never per-trial test score.

    A summary is withheld for an incomplete/failed grid. Standard deviations are
    across seeds; per-image test CIs remain in the raw CSV and are not averaged
    into an incorrectly labeled aggregate confidence interval.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[(row["head"], row["layer_group"], row["optimizer"])].append(row)
    summary = []
    for (head, group, optimizer), candidates in groups.items():
        record = {
            "head": head,
            "layer_group": group,
            "optimizer": optimizer,
            "dataset": candidates[0]["dataset"],
            "model": candidates[0]["model"],
            "layers": candidates[0]["layers"],
            "layer_count": candidates[0]["layer_count"],
            "selection_status": "incomplete_grid",
            "selected_lr": None,
            "seeds": None,
            "parameter_count": None,
            "mean_best_val_miou": None,
            "mean_test_miou": None,
            "std_test_miou": None,
            "mean_optimization_seconds": None,
            "mean_training_wall_seconds": None,
            "mean_selected_optimization_seconds": None,
            "converged_seeds": None,
            "stationary_seeds": None,
            "mixed_hardware_timings": None,
        }
        if any(row["status"] not in ("converged", "not_converged") for row in candidates):
            summary.append(record)
            continue
        rates = defaultdict(list)
        for row in candidates:
            rates[row["lr"]].append(row)
        seed_sets = [sorted(row["seed"] for row in values) for values in rates.values()]
        if any(seeds != seed_sets[0] or len(set(seeds)) != len(seeds) for seeds in seed_sets):
            raise ValueError("LR selection requires identical unique paired seeds.")
        lr = min(
            rates,
            key=lambda rate: (-statistics.mean(row["best_val_miou"] for row in rates[rate]), rate),
        )
        chosen = rates[lr]
        hardware_types = {
            features.identity({k: v for k, v in device.items() if k != "device"})
            for row in chosen
            for device in json.loads(row["hardware_history"])
        }
        record.update(
            selection_status="validation_selected",
            selected_lr=lr,
            seeds=json.dumps(seed_sets[0]),
            parameter_count=chosen[0]["parameter_count"],
            mean_best_val_miou=statistics.mean(row["best_val_miou"] for row in chosen),
            mean_test_miou=statistics.mean(row["test_miou"] for row in chosen),
            std_test_miou=statistics.stdev(row["test_miou"] for row in chosen)
            if len(chosen) > 1
            else 0.0,
            mean_optimization_seconds=statistics.mean(
                row["optimization_seconds"] for row in chosen
            ),
            mean_training_wall_seconds=statistics.mean(
                row["training_wall_seconds"] for row in chosen
            ),
            mean_selected_optimization_seconds=statistics.mean(
                row["selected_optimization_seconds"] for row in chosen
            ),
            converged_seeds=sum(row["status"] == "converged" for row in chosen),
            stationary_seeds=sum(row["stationary"] for row in chosen),
            mixed_hardware_timings=len(hardware_types) > 1
            or any(row["mixed_hardware_timings"] for row in chosen),
        )
        summary.append(record)
    return summary


def atomic_csv(path: Path, rows: list[dict]) -> None:
    """Replace a complete CSV under the caller's aggregate lock."""
    with features.atomic_stream(path, "w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output: Path, jobs: list[Job], metadata: dict) -> None:
    """Regenerate raw/selected CSVs from immutable results, retaining incomplete rows."""
    with FileLock(str(output / "aggregate.lock")):
        rows = []
        for job in jobs:
            expected = job_identity(job, metadata)
            directory = convergence.trial_directory(output / "trials", expected)
            result = (
                convergence.completed_result(directory, expected)
                if (directory / "result.json").exists()
                else None
            )
            rows.append(result_row(job, metadata, result))
        atomic_csv(output / "combined.csv", rows)
        atomic_csv(output / "validation_selected.csv", validation_summary(rows))


class StudyRunner(BaseGpuRunner):
    """Schedule one subprocess per trial, with a global Adam-before-L-BFGS barrier."""

    config: StudyConfig

    def __init__(self, config: StudyConfig, jobs: list[Job], spec: dict, metadata: dict) -> None:
        super().__init__(config, jobs)
        self.spec, self.metadata = spec, metadata

    def _summary_extra(self) -> dict:
        return {"output_dir": str(self.config.output_dir), "cache": str(self.config.cache)}

    def _command(self, job: Job, gpu: int, attempt: int) -> list[str]:  # noqa: ARG002
        command = [
            sys.executable,
            str(self.config.script),
            "--worker",
            str(self.config.state_dir / "jobs" / f"{job.job_id}.json"),
            "--device",
            f"cuda:{gpu}",
            "--cache",
            str(self.config.cache),
            "--output-dir",
            str(self.config.output_dir),
        ]
        if self.config.resume:
            command.append("--resume")
        return command

    def _is_complete(self, job: Job) -> bool:
        expected = job_identity(job, self.metadata)
        directory = convergence.trial_directory(self.config.output_dir / "trials", expected)
        if not (directory / "result.json").exists():
            return False
        result = convergence.completed_result(directory, expected)
        if result["status"] == "failed":
            raise RuntimeError(
                f"Immutable failed trial {directory}; inspect diagnostics and use a new study output."
            )
        return True

    def _run_job(self, job: Job, gpu: int) -> bool:
        success = self._run_attempt(job, gpu, 1) and self._is_complete(job)
        aggregate(self.config.output_dir, self.jobs, self.metadata)
        return success

    def _work_gpu(self, gpu: int, jobs: queue.Queue) -> None:
        while not self.stop_requested.is_set():
            try:
                job = jobs.get_nowait()
            except queue.Empty:  # allow-except: this worker has no more queued jobs
                return
            success = False
            with self.state_lock:
                self.counts["queued"] -= 1
                self.counts["running"] += 1
                self._write_summary_locked()
            try:
                success = self._run_job(job, gpu)
            finally:
                with self.state_lock:
                    self.counts["running"] -= 1
                    self.counts["completed" if success else "failed"] += 1
                    if not success:
                        self.failed_jobs.append(self._failed_record(job, gpu))
                    self._write_summary_locked()
                jobs.task_done()

    def _dispatch(self, pending: list[Job]) -> None:
        jobs: queue.Queue = queue.Queue()
        for job in pending:
            jobs.put(job)
        self.counts["queued"] = len(pending)
        with ThreadPoolExecutor(max_workers=len(self.config.gpus)) as executor:
            futures = [executor.submit(self._work_gpu, gpu, jobs) for gpu in self.config.gpus]
            for future in futures:
                future.result()  # Unlike bare threads, worker exceptions reach the CLI.
        if self.counts["failed"] or self.stop_requested.is_set():
            raise RuntimeError("Study interrupted or trial failed; inspect state/logs and resume.")

    def run(self) -> None:
        """Write worker specifications, skip only verified results and dispatch both phases."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = self.config.state_dir / "jobs"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        for job in self.jobs:
            manifest = {
                "job": asdict(job),
                "cache_spec": self.spec,
                "cache_tensor_sha256": self.metadata["tensor_sha256"],
            }
            path = manifest_dir / f"{job.job_id}.json"
            if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
                raise ValueError("Worker manifest identity mismatch.")
            features.atomic_json(path, manifest)
        try:
            for optimizer in ("adam", "lbfgs"):
                phase = [job for job in self.jobs if job.trial.optimizer == optimizer]
                pending = []
                for job in phase:
                    if self._is_complete(job):
                        self.counts["skipped_existing"] += 1
                    else:
                        pending.append(job)
                self._dispatch(pending)
            missing = [job.job_id for job in self.jobs if not self._is_complete(job)]
            if missing:
                raise RuntimeError(f"Missing completed trials: {missing}")
        finally:
            aggregate(self.config.output_dir, self.jobs, self.metadata)
            with self.state_lock:
                self._write_summary_locked()


def run_worker(args: argparse.Namespace) -> None:
    """Execute exactly one validated manifest using the same production script."""
    manifest = json.loads(args.worker.read_text())
    spec = manifest["cache_spec"]
    job_data = manifest["job"]
    trial = convergence.TrialConfig(**job_data["trial"])
    features.check_head_layers(trial.head, job_data["layers"])
    features.preflight_heads([trial.head])
    if spec["versions"] != features.software_versions():
        raise ValueError("Worker source/dependency identity changed after scheduling.")
    device = torch.device(args.device)
    features.configure_device(device, strict=spec["strict_determinism"])
    payload = features.load_cache(args.cache, spec)
    if payload["metadata"]["tensor_sha256"] != manifest["cache_tensor_sha256"]:
        raise ValueError("Worker cache checksum differs from scheduled cache.")
    selected, metadata = features.select_layers(payload, job_data["layers"])
    caches = {split: features.device_cache(data, device) for split, data in selected.items()}
    result = convergence.run_trial(
        trial, caches, metadata, args.output_dir / "trials", resume=args.resume
    )
    if result["status"] == "failed":
        raise features.DivergedError(result["failure"])


def build_parser(suite: str, doc: str) -> argparse.ArgumentParser:
    """Expose study controls without a time budget or implicit safety caps."""
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default="burn_scars")
    parser.add_argument("--model", default="timm/vit/vit_small_patch16_224")
    parser.add_argument("--bands", choices=("rgb", "all"), default="rgb")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--normalization",
        default="model_native",
        choices=("model_native", "bandspec_zscore", "minmax", "minmax_zscore", "identity"),
    )
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--gpus", default="all", help="all (default) or unique visible indices, e.g. 0,2."
    )
    parser.add_argument(
        "--cache",
        type=Path,
        help="Default: OUTPUT/features.pt; exact compatible union caches may be reused.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(f"results/segmentation_{suite}_study")
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        choices=features.HEADS,
        default=list(features.HEADS) if suite == "optimizer" else ["linear", "conv_block", "fpn"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--adam-lrs",
        nargs="+",
        type=float,
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
        if suite == "optimizer"
        else [1e-4, 1e-3, 1e-2],
    )
    parser.add_argument(
        "--lbfgs-lrs",
        nargs="+",
        type=float,
        default=[0.3, 1.0],
        help=argparse.SUPPRESS if suite == "layer" else None,
    )
    parser.add_argument(
        "--phase",
        choices=("adam", "lbfgs", "both") if suite == "optimizer" else ("adam",),
        default="both" if suite == "optimizer" else "adam",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--check-every", type=int, default=5, help="Adam epochs between full-objective checks."
    )
    parser.add_argument(
        "--lbfgs-iterations",
        type=int,
        default=20,
        help="Internal L-BFGS iterations per monitored chunk.",
    )
    parser.add_argument("--history-size", type=int, default=10)
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Historical-best patience and recent-window checks (at least two for the window).",
    )
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=50,
        help="Start collecting recent-loss checks at this many Adam epochs / L-BFGS iterations.",
    )
    parser.add_argument("--absolute-tol", type=float, default=1e-5)
    parser.add_argument("--relative-tol", type=float, default=1e-4)
    parser.add_argument("--gradient-tol", type=float, default=1e-7)
    parser.add_argument("--numerical-patience", type=int, default=3)
    parser.add_argument(
        "--max-iterations", type=int, help="Optional safety cap: never implies convergence."
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log grid and subprocess commands; no data, downloads, cache or output writes.",
    )
    parser.add_argument("--strict-determinism", action=argparse.BooleanOptionalAction, default=True)
    if suite == "optimizer":
        parser.add_argument(
            "--layers",
            nargs="+",
            default=LAYER_PRESETS["spread_four"],
            help="Ordered deepest-first connections; patch_linear explicitly uses only the first.",
        )
    else:
        groups = parser.add_mutually_exclusive_group()
        groups.add_argument(
            "--layer-groups", nargs="+", choices=list(LAYER_PRESETS), default=list(LAYER_PRESETS)
        )
        groups.add_argument(
            "--layer-groups-json",
            help='Custom ordered connections, e.g. \'{"deep":["blocks.11"]}\'.',
        )
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser


def main(suite: str, doc: str, argv: list[str] | None = None) -> None:
    """Prepare a single union cache and schedule independent, restartable trial workers."""
    args = build_parser(suite, doc).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.worker:
        run_worker(args)
        return
    if args.num_workers < 0:
        raise ValueError("num_workers must be nonnegative.")
    jobs = build_jobs(args, suite)
    features.preflight_heads([] if args.dry_run else args.heads)
    gpus = selected_gpus(args.gpus, dry_run=args.dry_run)
    extraction = features.ExtractionConfig(
        **{name: getattr(args, name) for name in features.ExtractionConfig.__dataclass_fields__}
    )
    spec = features.cache_spec(extraction, layer_union(jobs))
    output = resolve_path(ROOT, args.output_dir)
    cache = resolve_path(ROOT, args.cache) if args.cache else output / "features.pt"
    config = StudyConfig(
        root=ROOT,
        cli=Path(sys.executable),
        state_dir=output / "state",
        gpus=gpus,
        num_workers=args.num_workers,
        max_attempts=1,
        script=Path(__file__).with_name(f"run_segmentation_{suite}_study.py"),
        cache=cache,
        output_dir=output,
        resume=args.resume,
    )
    if args.dry_run:
        runner = StudyRunner(config, jobs, spec, {})
        logger.info("%d trials; extract one union cache for layers %s", len(jobs), spec["layers"])
        for index, job in enumerate(jobs):
            logger.info(
                "%s %s: %s",
                job.layer_group,
                asdict(job.trial),
                runner._command(job, gpus[index % len(gpus)], 1),
            )
        return
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / "study.lock"), timeout=0):
        path = output / "metadata.json"
        expected = json.loads(
            json.dumps({"suite": suite, "cache_spec": spec, "jobs": [asdict(job) for job in jobs]})
        )
        if path.exists():
            if not args.resume:
                raise FileExistsError("Study exists; use --resume or a new output directory.")
            if json.loads(path.read_text()) != expected:
                raise ValueError("Incompatible study configuration; refusing resume.")
        else:
            if (
                (output / "trials").exists()
                or (output / "state").exists()
                or (output / "combined.csv").exists()
            ):
                raise ValueError("Study metadata missing for existing artifacts.")
            features.atomic_json(path, expected)
        payload = features.prepare_cache(
            cache, spec, torch.device(f"cuda:{gpus[0]}"), args.num_workers
        )
        metadata = payload["metadata"]
        del payload
        torch.cuda.empty_cache()
        runner = StudyRunner(config, jobs, spec, metadata)
        previous_handlers = {
            sig: signal.signal(sig, runner.request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            runner.run()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
