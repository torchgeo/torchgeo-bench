#!/usr/bin/env python3
"""Run a cross-model segmentation training-protocol sensitivity study."""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)
VAL_PATTERN = re.compile(r"Epoch (\d+) Val mIoU: (\S+)")


@dataclass(frozen=True)
class Model:
    """Model preset and conservative cache-extraction batch size."""

    config: str
    name: str
    loader_batch_size: int
    probe_batch_size: int


@dataclass(frozen=True)
class Variant:
    """Segmentation-head optimization configuration."""

    name: str
    epochs: int
    lr: float
    scheduler: str


@dataclass(frozen=True)
class Dataset:
    """Dataset and selected input-band mode."""

    name: str
    bands: str


@dataclass(frozen=True)
class Job:
    """One model, dataset, and optimization-variant combination."""

    model: Model
    dataset: Dataset
    variant: Variant

    @property
    def job_id(self) -> str:
        """Return a filesystem-safe identifier."""
        return f"{self.model.name}__{self.dataset.name}__{self.variant.name}"


MODELS = [
    Model("timm/convnext_large_dinov3", "convnext_large_dinov3", 8, 8),
    Model("timm/resnet50", "resnet50", 32, 8),
    Model("timm/efficientnet_b0", "efficientnet_b0", 32, 8),
    Model("timm/densenet121", "densenet121", 16, 8),
    Model("timm/maxvit_tiny_tf_224", "maxvit_tiny_tf_224", 12, 8),
    Model("torchgeo/scalemae_large_fmow", "tgeo_scalemae_large_fmow", 8, 8),
    Model(
        "torchgeo/swinv2b_s2rgb_satlas_mi",
        "tgeo_swinv2b_s2rgb_satlas_mi",
        8,
        8,
    ),
]
DATASETS = [
    Dataset("burn_scars", "all"),
    Dataset("cloudsen12", "rgb"),
    Dataset("dynamic_earthnet", "rgb"),
    Dataset("pastis", "rgb"),
]
VARIANTS = [
    Variant("baseline_10ep_cosine_1e-3", 10, 1e-3, "cosine"),
    Variant("long_30ep_cosine_1e-3", 30, 1e-3, "cosine"),
    Variant("long_50ep_cosine_1e-3", 50, 1e-3, "cosine"),
    Variant("low_lr_30ep_cosine_3e-4", 30, 3e-4, "cosine"),
    Variant("high_lr_30ep_cosine_3e-3", 30, 3e-3, "cosine"),
    Variant("constant_30ep_1e-3", 30, 1e-3, "none"),
]


def build_jobs() -> list[Job]:
    """Return the complete cross-model protocol-study matrix."""
    return [
        Job(model, dataset, variant)
        for model in MODELS
        for dataset in DATASETS
        for variant in VARIANTS
    ]


def _source_hash(root: Path) -> str:
    hasher = hashlib.sha256()
    paths = sorted((root / "src/torchgeo_bench").rglob("*.py"))
    paths.extend(sorted((root / "src/torchgeo_bench/conf").rglob("*.yaml")))
    for path in paths:
        hasher.update(str(path.relative_to(root)).encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def study_metadata(root: Path) -> dict[str, object]:
    """Return the result-affecting study configuration."""
    return {
        "schema_version": 1,
        "source_hash": _source_hash(root),
        "models": [asdict(model) for model in MODELS],
        "datasets": [asdict(dataset) for dataset in DATASETS],
        "variants": [asdict(variant) for variant in VARIANTS],
        "head_type": "fpn",
        "image_size": 224,
        "normalization": "bandspec_zscore",
        "partition": "default",
        "seed": 0,
        "cache_features": True,
        "cache_dtype": "float16",
    }


@dataclass(frozen=True)
class StudyConfig:
    """Runtime paths and resource controls."""

    root: Path
    cli: Path
    raw_dir: Path
    state_dir: Path
    combined_output: Path
    gpus: list[int]
    num_workers: int
    max_attempts: int


class StudyRunner:
    """Dynamically schedule protocol-study jobs across GPUs."""

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.jobs = build_jobs()
        self.log_dir = config.state_dir / "logs"
        self.events_path = config.state_dir / "events.jsonl"
        self.summary_path = config.state_dir / "summary.json"
        self.failed_path = config.state_dir / "failed_jobs.json"
        self.metadata_path = config.raw_dir.parent / "metadata.json"
        self.state_lock = threading.Lock()
        self.failed_jobs: list[dict] = []
        self.counts = {
            "total": len(self.jobs),
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped_existing": 0,
        }

    def _output_path(self, job: Job) -> Path:
        return self.config.raw_dir / f"{job.job_id}.csv"

    def _validate_metadata(self) -> None:
        expected = study_metadata(self.config.root)
        if self.metadata_path.exists():
            if json.loads(self.metadata_path.read_text()) != expected:
                raise RuntimeError(f"Incompatible study metadata at {self.metadata_path}.")
            return
        existing_outputs = list(self.config.raw_dir.glob("*.csv"))
        if existing_outputs:
            raise RuntimeError(
                f"{self.metadata_path} is missing but {len(existing_outputs)} raw outputs exist; "
                "refusing to assign current metadata to results of unknown provenance."
            )
        temporary = self.metadata_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.metadata_path)

    def _is_complete(self, job: Job) -> bool:
        path = self._output_path(job)
        if not path.exists() or path.stat().st_size == 0:
            return False
        with path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != 1:
            return False
        row = rows[0]
        expected = {
            "dataset": job.dataset.name,
            "method": "seg-fpn",
            "name": job.model.name,
            "bands": job.dataset.bands,
            "image_size": "224",
        }
        if any(row.get(key) != value for key, value in expected.items()):
            return False
        try:
            valid_csv = (
                math.isfinite(float(row["metric_value"]))
                and float(row["best_lr"]) == job.variant.lr
                and int(float(row["best_batch_size"])) == job.model.probe_batch_size
            )
        except (KeyError, TypeError, ValueError):
            return False
        if not valid_csv:
            return False
        valid_logs = []
        for log_path in self.log_dir.glob(f"{job.job_id}.attempt*.log"):
            text = log_path.read_text(errors="replace")
            matches = [(int(epoch), float(value)) for epoch, value in VAL_PATTERN.findall(text)]
            if (
                [epoch for epoch, _ in matches] == list(range(1, job.variant.epochs + 1))
                and all(math.isfinite(value) for _, value in matches)
                and "Benchmark complete." in text
            ):
                valid_logs.append(log_path)
        return len(valid_logs) == 1

    def _write_summary_locked(self) -> None:
        payload = {
            **self.counts,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "raw_dir": str(self.config.raw_dir),
            "combined_output": str(self.config.combined_output),
            "gpus": self.config.gpus,
        }
        temporary = self.summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.summary_path)
        self.failed_path.write_text(json.dumps(self.failed_jobs, indent=2, sort_keys=True) + "\n")

    def _write_event(self, event: str, job: Job, **details: object) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "job_id": job.job_id,
            **details,
        }
        with self.state_lock, self.events_path.open("a") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def _command(self, job: Job, gpu: int, attempt: int) -> list[str]:
        loader_batch = max(1, job.model.loader_batch_size // (2 ** (attempt - 1)))
        return [
            str(self.config.cli),
            "run",
            f"model={job.model.config}",
            f"dataset.names=[{job.dataset.name}]",
            f"dataset.bands={job.dataset.bands}",
            "dataset.image_size=224",
            f"dataset.batch_size={loader_batch}",
            f"dataset.num_workers={self.config.num_workers}",
            f"device=cuda:{gpu}",
            "eval.segmentation.head_type=fpn",
            f"eval.segmentation.epochs={job.variant.epochs}",
            f"eval.segmentation.lr={job.variant.lr}",
            f"eval.segmentation.lr_scheduler={job.variant.scheduler}",
            f"eval.segmentation.batch_size={job.model.probe_batch_size}",
            "eval.segmentation.cache_features=true",
            "eval.segmentation.cache_dtype=float16",
            "verbose=true",
            f"output={self._output_path(job)}",
            "resume=false",
        ]

    def _run_job(self, job: Job, gpu: int) -> bool:
        output_path = self._output_path(job)
        if output_path.exists() and not self._is_complete(job):
            output_path.unlink()
        for stale_log in self.log_dir.glob(f"{job.job_id}.attempt*.log"):
            stale_log.unlink()
        for attempt in range(1, self.config.max_attempts + 1):
            command = self._command(job, gpu, attempt)
            log_path = self.log_dir / f"{job.job_id}.attempt{attempt}.log"
            self._write_event(
                "started",
                job,
                gpu=gpu,
                attempt=attempt,
                command=command,
                log=str(log_path),
            )
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "4"
            environment["MKL_NUM_THREADS"] = "4"
            started = time.monotonic()
            with log_path.open("w") as log:
                result = subprocess.run(
                    command,
                    cwd=self.config.root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.monotonic() - started
            self._write_event(
                "completed" if result.returncode == 0 else "attempt_failed",
                job,
                gpu=gpu,
                attempt=attempt,
                returncode=result.returncode,
                elapsed_seconds=round(elapsed, 3),
                log=str(log_path),
            )
            if result.returncode == 0 and self._is_complete(job):
                return True
            output_path.unlink(missing_ok=True)
        return False

    def _worker(self, gpu: int, jobs: queue.Queue[Job]) -> None:
        while True:
            try:
                job = jobs.get_nowait()
            except queue.Empty:
                return
            with self.state_lock:
                self.counts["queued"] -= 1
                self.counts["running"] += 1
                self._write_summary_locked()
            success = self._run_job(job, gpu)
            with self.state_lock:
                self.counts["running"] -= 1
                self.counts["completed" if success else "failed"] += 1
                if not success:
                    self.failed_jobs.append(asdict(job))
                self._write_summary_locked()
                logger.info(
                    "Progress: %d/%d completed, %d failed, %d running",
                    self.counts["completed"],
                    self.counts["total"],
                    self.counts["failed"],
                    self.counts["running"],
                )
            jobs.task_done()

    def run(self) -> None:
        """Run pending jobs and generate the combined analysis CSV."""
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_metadata()

        pending: list[Job] = []
        for job in self.jobs:
            if self._is_complete(job):
                self.counts["skipped_existing"] += 1
            else:
                pending.append(job)
        self.counts["queued"] = len(pending)
        jobs: queue.Queue[Job] = queue.Queue()
        for job in pending:
            jobs.put(job)
        with self.state_lock:
            self._write_summary_locked()

        workers = [
            threading.Thread(target=self._worker, args=(gpu, jobs), name=f"gpu-{gpu}")
            for gpu in self.config.gpus
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()

        missing = [job.job_id for job in self.jobs if not self._is_complete(job)]
        if self.counts["failed"] or missing:
            raise RuntimeError(
                f"Study incomplete: {self.counts['failed']} failures and "
                f"{len(missing)} missing outputs. See {self.failed_path}."
            )
        analyzer = Path(__file__).with_name("analyze_segmentation_protocol_study.py")
        subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--raw-dir",
                str(self.config.raw_dir),
                "--log-dir",
                str(self.log_dir),
                "--events",
                str(self.events_path),
                "--output",
                str(self.config.combined_output),
            ],
            cwd=self.config.root,
            check=True,
        )


def _parse_gpus(value: str) -> list[int]:
    available = torch.cuda.device_count()
    gpus = list(range(available)) if value == "all" else [int(item) for item in value.split(",")]
    if not gpus:
        raise ValueError("No GPUs selected.")
    invalid = [gpu for gpu in gpus if gpu < 0 or gpu >= available]
    if invalid:
        raise ValueError(f"Invalid GPU indices {invalid}; {available} devices are visible.")
    return gpus


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--raw-dir", type=Path, default=Path("results/segmentation_protocol/raw"))
    parser.add_argument(
        "--state-dir", type=Path, default=Path("results/segmentation_protocol/state")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/segmentation_protocol_study.csv")
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run the protocol sensitivity study."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    config = StudyConfig(
        root=root,
        cli=Path(sys.executable).with_name("torchgeo-bench"),
        raw_dir=_resolve(root, args.raw_dir),
        state_dir=_resolve(root, args.state_dir),
        combined_output=_resolve(root, args.output),
        gpus=_parse_gpus(args.gpus),
        num_workers=args.num_workers,
        max_attempts=args.max_attempts,
    )
    logger.info(
        "Study: %d models x %d datasets x %d variants = %d jobs",
        len(MODELS),
        len(DATASETS),
        len(VARIANTS),
        len(build_jobs()),
    )
    if args.dry_run:
        return

    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    output_lock = FileLock(str(config.raw_dir.parent / "runner.lock"))
    state_lock = FileLock(str(config.state_dir / "runner.lock"))
    try:
        with output_lock.acquire(timeout=0), state_lock.acquire(timeout=0):
            StudyRunner(config).run()
    except Timeout as error:
        raise RuntimeError(
            "Another protocol study is using the output or state directory."
        ) from error


if __name__ == "__main__":
    main()
