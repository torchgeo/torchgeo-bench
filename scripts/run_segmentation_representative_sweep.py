#!/usr/bin/env python3
"""Run the representative segmentation benchmark matrix across all GPUs."""

import argparse
import csv
import json
import logging
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from filelock import FileLock, Timeout

logger = logging.getLogger(__name__)

DATASETS = [
    "burn_scars",
    "caffe",
    "cloudsen12",
    "dynamic_earthnet",
    "flair2",
    "fotw",
    "kuro_siwo",
    "pastis",
    "spacenet2",
    "spacenet7",
]
HEADS = ["linear", "conv_block", "fpn", "dpt"]
BANDS = ["rgb", "all"]
EPOCHS = 10


@dataclass(frozen=True)
class Model:
    """Model preset and conservative per-process batch sizes."""

    config: str
    name: str
    loader_batch_size: int
    probe_batch_size: int


@dataclass(frozen=True)
class Job:
    """One model, dataset, probe-head, and band-selection combination."""

    model: Model
    dataset: str
    head: str
    bands: str

    @property
    def job_id(self) -> str:
        """Return a filesystem-safe identifier."""
        model = self.model.config.replace("/", "__")
        return f"{model}__{self.dataset}__{self.head}__{self.bands}"

    @property
    def method(self) -> str:
        """Return the method label written by torchgeo-bench."""
        return f"seg-{self.head}"


MODELS = [
    Model("timm/resnet50", "resnet50", 32, 16),
    Model("timm/convnext_tiny", "convnext_tiny", 24, 16),
    Model("timm/convnext_large_dinov3", "convnext_large_dinov3", 8, 8),
    Model("timm/efficientnet_b0", "efficientnet_b0", 32, 16),
    Model("timm/densenet121", "densenet121", 16, 16),
    Model("timm/maxvit_tiny_tf_224", "maxvit_tiny_tf_224", 12, 12),
    Model("torchgeo/scalemae_large_fmow", "tgeo_scalemae_large_fmow", 8, 8),
    Model(
        "torchgeo/swinv2b_s2rgb_satlas_mi",
        "tgeo_swinv2b_s2rgb_satlas_mi",
        8,
        8,
    ),
]

# These inputs violate the pretrained wrapper's normalization contract,
# independent of probe head or batch size.
UNSUPPORTED_INPUTS = {
    ("torchgeo/scalemae_large_fmow", "caffe", "rgb"),
    ("torchgeo/scalemae_large_fmow", "caffe", "all"),
    ("torchgeo/scalemae_large_fmow", "pastis", "all"),
    ("torchgeo/swinv2b_s2rgb_satlas_mi", "pastis", "all"),
}


def build_jobs() -> list[Job]:
    """Return every supported combination in the representative matrix."""
    jobs: list[Job] = []
    for model in MODELS:
        for dataset in DATASETS:
            for head in HEADS:
                for bands in BANDS:
                    job = Job(model, dataset, head, bands)
                    if (model.config, dataset, bands) not in UNSUPPORTED_INPUTS:
                        jobs.append(job)
    return jobs


def sweep_metadata(image_size: int) -> dict[str, object]:
    """Return the result-affecting configuration fingerprint."""
    return {
        "schema_version": 1,
        "epochs": EPOCHS,
        "image_size": image_size,
        "seed": 0,
        "normalization": "bandspec_zscore",
        "interpolation": "bilinear",
        "partition": "default",
        "cache_features": True,
        "cache_dtype": "float16",
        "models": [asdict(model) for model in MODELS],
        "datasets": DATASETS,
        "heads": HEADS,
        "bands": BANDS,
        "unsupported_inputs": [list(item) for item in sorted(UNSUPPORTED_INPUTS)],
    }


@dataclass(frozen=True)
class SweepConfig:
    """Runtime configuration for the multi-GPU sweep."""

    root: Path
    cli: Path
    output: Path
    state_dir: Path
    gpus: list[int]
    image_size: int
    num_workers: int
    max_attempts: int
    seed: int


class SweepRunner:
    """Dynamically schedule independent benchmark jobs across GPUs."""

    def __init__(self, config: SweepConfig) -> None:
        self.config = config
        self.log_dir = config.state_dir / "logs"
        self.events_path = config.state_dir / "events.jsonl"
        self.summary_path = config.state_dir / "summary.json"
        self.failed_path = config.state_dir / "failed_jobs.json"
        self.metadata_path = Path(f"{config.output}.sweep.json")
        self.stop_requested = threading.Event()
        self.state_lock = threading.Lock()
        self.jobs = build_jobs()
        self.failed_jobs: list[dict] = []
        self.counts = {
            "candidate_total": len(MODELS) * len(DATASETS) * len(HEADS) * len(BANDS),
            "unsupported": len(UNSUPPORTED_INPUTS) * len(HEADS),
            "total": len(self.jobs),
            "queued": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "skipped_existing": 0,
        }

    @staticmethod
    def _normalized_size(value: str) -> str:
        try:
            return str(int(float(value)))
        except ValueError:
            return value

    def _completed_keys(self) -> set[tuple[str, str, str, str, str]]:
        if not self.config.output.exists() or self.config.output.stat().st_size == 0:
            return set()
        with self.config.output.open(newline="") as file:
            rows = csv.DictReader(file)
            return {
                (
                    row.get("dataset", ""),
                    row.get("method", ""),
                    row.get("name", ""),
                    row.get("bands", ""),
                    self._normalized_size(row.get("image_size", "")),
                )
                for row in rows
            }

    def _result_key(self, job: Job) -> tuple[str, str, str, str, str]:
        return (
            job.dataset,
            job.method,
            job.model.name,
            job.bands,
            str(self.config.image_size),
        )

    def _validate_metadata(self) -> None:
        expected = sweep_metadata(self.config.image_size)
        if self.config.output.exists() and self.config.output.stat().st_size > 0:
            if not self.metadata_path.exists():
                raise RuntimeError(
                    f"{self.config.output} exists without {self.metadata_path}; "
                    "refusing to resume results with an unknown configuration."
                )
            actual = json.loads(self.metadata_path.read_text())
            if actual != expected:
                raise RuntimeError(
                    f"{self.config.output} was created by an incompatible sweep configuration."
                )
            return
        temporary = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")
        temporary.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        temporary.replace(self.metadata_path)

    def _write_event(self, event: str, job: Job | None, **details: object) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event": event,
            "job_id": job.job_id if job else None,
            **details,
        }
        with self.state_lock, self.events_path.open("a") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")

    def _write_summary_locked(self) -> None:
        payload = {
            **self.counts,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": str(self.config.output),
            "failed_jobs": str(self.failed_path),
            "gpus": self.config.gpus,
            "epochs": EPOCHS,
            "image_size": self.config.image_size,
        }
        temp = self.summary_path.with_suffix(".tmp")
        temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temp.replace(self.summary_path)
        self.failed_path.write_text(json.dumps(self.failed_jobs, indent=2, sort_keys=True) + "\n")

    def _command(self, job: Job, gpu: int, attempt: int) -> list[str]:
        divisor = 2 ** (attempt - 1)
        loader_batch = max(1, job.model.loader_batch_size // divisor)
        probe_batch = job.model.probe_batch_size
        return [
            str(self.config.cli),
            "run",
            f"model={job.model.config}",
            f"dataset.names=[{job.dataset}]",
            f"dataset.bands={job.bands}",
            f"dataset.image_size={self.config.image_size}",
            f"dataset.batch_size={loader_batch}",
            f"dataset.num_workers={self.config.num_workers}",
            f"device=cuda:{gpu}",
            f"eval.segmentation.head_type={job.head}",
            f"eval.segmentation.epochs={EPOCHS}",
            f"eval.segmentation.batch_size={probe_batch}",
            "eval.segmentation.cache_features=true",
            "eval.segmentation.cache_dtype=float16",
            f"output={self.config.output}",
            "resume=true",
        ]

    def _run_job(self, job: Job, gpu: int) -> bool:
        for attempt in range(1, self.config.max_attempts + 1):
            log_path = self.log_dir / f"{job.job_id}.attempt{attempt}.log"
            command = self._command(job, gpu, attempt)
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = "4"
            environment["MKL_NUM_THREADS"] = "4"
            environment["TOKENIZERS_PARALLELISM"] = "false"
            self._write_event(
                "started",
                job,
                gpu=gpu,
                attempt=attempt,
                command=command,
                log=str(log_path),
            )
            started = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.run(
                    command,
                    cwd=self.config.root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            elapsed = time.monotonic() - started
            if process.returncode == 0:
                self._write_event(
                    "completed",
                    job,
                    gpu=gpu,
                    attempt=attempt,
                    elapsed_seconds=round(elapsed, 3),
                    log=str(log_path),
                )
                return True
            self._write_event(
                "attempt_failed",
                job,
                gpu=gpu,
                attempt=attempt,
                returncode=process.returncode,
                elapsed_seconds=round(elapsed, 3),
                log=str(log_path),
            )
            if self.stop_requested.is_set():
                break
        return False

    def _worker(self, gpu: int, jobs: queue.Queue[Job]) -> None:
        while not self.stop_requested.is_set():
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
                if success:
                    self.counts["completed"] += 1
                else:
                    self.counts["failed"] += 1
                    self.failed_jobs.append(
                        {
                            **asdict(job),
                            "job_id": job.job_id,
                            "gpu": gpu,
                        }
                    )
                self._write_summary_locked()
                finished = self.counts["completed"] + self.counts["failed"]
                logger.info(
                    "Progress: %d/%d finished, %d failed, %d running",
                    finished,
                    self.counts["total"] - self.counts["skipped_existing"],
                    self.counts["failed"],
                    self.counts["running"],
                )
            jobs.task_done()

    def request_stop(self, signum: int, _frame: object) -> None:
        """Stop assigning jobs after currently running commands finish."""
        self.stop_requested.set()
        self._write_event("stop_requested", None, signal=signum)

    def run(self) -> None:
        """Run all pending combinations and update state after every job."""
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_metadata()

        existing = self._completed_keys()
        pending: list[Job] = []
        for job in self.jobs:
            if self._result_key(job) in existing:
                self.counts["skipped_existing"] += 1
            else:
                pending.append(job)

        random.Random(self.config.seed).shuffle(pending)
        jobs: queue.Queue[Job] = queue.Queue()
        for job in pending:
            jobs.put(job)
        self.counts["queued"] = len(pending)
        with self.state_lock:
            self._write_summary_locked()
        self._write_event("sweep_started", None, pending=len(pending), total=self.counts["total"])

        workers = [
            threading.Thread(target=self._worker, args=(gpu, jobs), name=f"gpu-{gpu}")
            for gpu in self.config.gpus
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()

        with self.state_lock:
            self._write_summary_locked()
        self._write_event(
            "sweep_finished",
            None,
            completed=self.counts["completed"],
            failed=self.counts["failed"],
            queued=self.counts["queued"],
        )
        missing = {self._result_key(job) for job in self.jobs} - self._completed_keys()
        if self.counts["failed"] or missing:
            raise RuntimeError(
                f"Sweep incomplete: {self.counts['failed']} failed jobs and "
                f"{len(missing)} missing result rows. See {self.failed_path}."
            )


def parse_gpus(value: str) -> list[int]:
    """Parse ``all`` or a comma-separated list of visible CUDA device indices."""
    available = torch.cuda.device_count()
    if value == "all":
        gpus = list(range(available))
    else:
        gpus = [int(item) for item in value.split(",") if item]
    if not gpus:
        raise ValueError("No GPUs selected.")
    invalid = [gpu for gpu in gpus if gpu < 0 or gpu >= available]
    if invalid:
        raise ValueError(
            f"GPU indices {invalid} are invalid; {available} CUDA devices are visible."
        )
    return gpus


def ensure_datasets(config: SweepConfig, *, download_missing: bool) -> None:
    """Download GeoBench V2 datasets whose canonical directories are absent."""
    data_root = config.root / "data/geobenchv2"
    missing = [name for name in DATASETS if not (data_root / name).is_dir()]
    if not missing:
        return
    if not download_missing:
        raise FileNotFoundError(f"Missing datasets: {', '.join(missing)}")
    logger.info("Downloading missing datasets: %s", ", ".join(missing))
    subprocess.run(
        [
            str(config.cli),
            "download",
            "geobench_v2",
            "--datasets",
            ",".join(missing),
        ],
        cwd=config.root,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="all", help="'all' or comma-separated GPU indices")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    return path if path.is_absolute() else root / path


def main() -> None:
    """Run the configured sweep."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cli = Path(sys.executable).with_name("torchgeo-bench")
    if not cli.is_file():
        raise FileNotFoundError(
            f"{cli} does not exist. Activate the torchgeo-bench environment first."
        )

    output = resolve_path(
        root,
        args.output
        or Path(f"results/segmentation_representative_rgb_all_{args.image_size}_{EPOCHS}ep.csv"),
    )
    state_dir = (
        resolve_path(root, args.state_dir)
        if args.state_dir is not None
        else Path(f"{output}.state")
    )
    config = SweepConfig(
        root=root,
        cli=cli,
        output=output,
        state_dir=state_dir,
        gpus=parse_gpus(args.gpus),
        image_size=args.image_size,
        num_workers=args.num_workers,
        max_attempts=args.max_attempts,
        seed=args.seed,
    )
    candidate_total = len(MODELS) * len(DATASETS) * len(HEADS) * len(BANDS)
    supported_total = len(build_jobs())
    logger.info(
        "Matrix: %d candidates, %d unsupported, %d supported jobs",
        candidate_total,
        candidate_total - supported_total,
        supported_total,
    )
    logger.info(
        "Dimensions: %d models x %d datasets x %d heads x %d band modes",
        len(MODELS),
        len(DATASETS),
        len(HEADS),
        len(BANDS),
    )
    logger.info("GPUs: %s", config.gpus)
    logger.info("Results: %s", config.output)
    if args.dry_run:
        return

    ensure_datasets(config, download_missing=not args.no_download)
    runner = SweepRunner(config)
    signal.signal(signal.SIGINT, runner.request_stop)
    signal.signal(signal.SIGTERM, runner.request_stop)
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    output_lock = FileLock(f"{config.output}.runner.lock")
    state_lock = FileLock(str(config.state_dir / "runner.lock"))
    try:
        with output_lock.acquire(timeout=0), state_lock.acquire(timeout=0):
            runner.run()
    except Timeout as error:
        raise RuntimeError(
            f"Another sweep is writing {config.output} or {config.state_dir}."
        ) from error


if __name__ == "__main__":
    main()
