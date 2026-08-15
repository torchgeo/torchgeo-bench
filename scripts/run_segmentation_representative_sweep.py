#!/usr/bin/env python3
"""Run the representative segmentation benchmark matrix across all GPUs."""

import argparse
import csv
import json
import logging
import random
import signal
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from _seg_sweep_common import (
    BaseGpuRunner,
    Model,
    RunnerConfig,
    parse_gpus,
    resolve_path,
    run_exclusively,
    torchgeo_bench_cli,
    write_json_atomic,
)

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
class SweepConfig(RunnerConfig):
    """Runtime configuration for the multi-GPU sweep."""

    output: Path
    image_size: int
    seed: int


class SweepRunner(BaseGpuRunner):
    """Dynamically schedule independent benchmark jobs across GPUs."""

    config: SweepConfig
    subprocess_env = BaseGpuRunner.subprocess_env | {"TOKENIZERS_PARALLELISM": "false"}

    def __init__(self, config: SweepConfig) -> None:
        super().__init__(config, build_jobs())
        self.metadata_path = Path(f"{config.output}.sweep.json")
        self.counts["candidate_total"] = len(MODELS) * len(DATASETS) * len(HEADS) * len(BANDS)
        self.counts["unsupported"] = len(UNSUPPORTED_INPUTS) * len(HEADS)

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
        write_json_atomic(self.metadata_path, expected)

    def _summary_extra(self) -> dict[str, object]:
        return {
            "results": str(self.config.output),
            "failed_jobs": str(self.failed_path),
            "epochs": EPOCHS,
            "image_size": self.config.image_size,
        }

    def _failed_record(self, job: Job, gpu: int) -> dict:
        return {**asdict(job), "job_id": job.job_id, "gpu": gpu}

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

    def run(self) -> None:
        """Run all pending combinations and update state after every job."""
        self.config.output.parent.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_metadata()

        existing = self._completed_keys()
        pending = self._partition_pending(lambda job: self._result_key(job) in existing)
        random.Random(self.config.seed).shuffle(pending)
        self._write_event("sweep_started", None, pending=len(pending), total=self.counts["total"])
        self._dispatch(pending)

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


def main() -> None:
    """Run the configured sweep."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    cli = torchgeo_bench_cli(require=True)

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
    run_exclusively(
        runner.run,
        [f"{config.output}.runner.lock", config.state_dir / "runner.lock"],
        f"Another sweep is writing {config.output} or {config.state_dir}.",
    )


if __name__ == "__main__":
    main()
