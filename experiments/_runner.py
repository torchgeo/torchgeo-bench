"""Shared GPU queue dispatcher for non-custom experiment runners.

Each non-custom experiment script builds a list of :class:`Job` instances
(one per ``torchgeo-bench run …`` invocation it wants to make) and calls
:func:`run_jobs` to execute them. With a single device the jobs run
sequentially; with multiple devices they fan out across one worker thread
per device, each pulling jobs from a shared queue.

This module is invoked as a sibling import from scripts in the same
directory (``from _runner import …``) — Python prepends the script's
directory to ``sys.path`` so no path setup is required.
"""

import argparse
import logging
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Job:
    """One ``torchgeo-bench run`` invocation.

    Attributes:
        label: Short human-readable identifier for log lines.
        overrides: Hydra-style overrides forwarded to ``torchgeo-bench run``
            (e.g. ``["model=timm/resnet18", "dataset.names=[m-eurosat]"]``).
            ``device`` and ``output`` are appended automatically by the
            runner — do not include them here.
    """

    label: str
    overrides: list[str] = field(default_factory=list)


@dataclass
class _JobResult:
    label: str
    gpu: int
    elapsed: float
    returncode: int
    stderr_tail: str = ""


def add_devices_argument(parser: argparse.ArgumentParser) -> None:
    """Register a ``--devices`` flag that takes one or more GPU indices.

    Defaults to ``[0]`` (single GPU, sequential execution).
    """
    parser.add_argument(
        "--devices",
        nargs="+",
        type=int,
        default=[0],
        metavar="GPU",
        help="One or more CUDA device indices (e.g. --devices 0 1 2). "
        "With a single device jobs run sequentially; with multiple devices "
        "jobs are dispatched via a queue with one worker per device. "
        "Default: 0.",
    )


def default_output(script_file: str | Path) -> str:
    """Derive the standard ``results/<basename>.csv`` path from a script's ``__file__``.

    Drops the ``run_`` prefix and ``.py`` suffix. For example,
    ``run_cls_token_experiment.py`` becomes ``results/cls_token_experiment.csv``.
    """
    stem = Path(script_file).stem.removeprefix("run_")
    return f"results/{stem}.csv"


def _run_one(job: Job, gpu: int, idx: int, total: int, output: str) -> _JobResult:
    """Shell out to ``torchgeo-bench run …`` for a single job."""
    cmd = [
        "torchgeo-bench",
        "run",
        *job.overrides,
        f"device=cuda:{gpu}",
        f"output={output}",
        "resume=true",
    ]

    logger.info("[%d/%d] START %s on cuda:%d", idx, total, job.label, gpu)
    start = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - start

    if proc.returncode == 0:
        logger.info("[%d/%d] DONE %s (%.0fs) on cuda:%d", idx, total, job.label, elapsed, gpu)
        return _JobResult(job.label, gpu, elapsed, 0)

    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    logger.error(
        "[%d/%d] FAILED %s (%.0fs) on cuda:%d\n%s",
        idx,
        total,
        job.label,
        elapsed,
        gpu,
        stderr_tail,
    )
    return _JobResult(job.label, gpu, elapsed, proc.returncode, stderr_tail)


def _worker(
    gpu: int,
    job_queue: "Queue[tuple[int, Job] | None]",
    total: int,
    output: str,
    results: "Queue[_JobResult]",
) -> None:
    """Pull jobs off the queue and run them on the assigned GPU until empty."""
    while (item := job_queue.get()) is not None:
        idx, job = item
        results.put(_run_one(job, gpu, idx, total, output))


def summarize_results(results: list[_JobResult], total: int, elapsed: float, output: str) -> int:
    """Log job timings and failures, returning the run exit code."""
    passed = sum(1 for r in results if r.returncode == 0)
    failed = total - passed

    logger.info(
        "Run complete: %d/%d passed, %d failed, %.0fs elapsed; results in %s",
        passed,
        total,
        failed,
        elapsed,
        output,
    )

    if failed:
        for r in results:
            if r.returncode != 0:
                logger.error("Failed job %s (%.0fs, cuda:%d)", r.label, r.elapsed, r.gpu)

    if passed:
        times = sorted(
            [(r.label, r.elapsed) for r in results if r.returncode == 0],
            key=lambda x: x[1],
        )
        avg = sum(t for _, t in times) / len(times)
        logger.info("Average time per job: %.0fs", avg)
        logger.info("Fastest: %s (%.0fs)", times[0][0], times[0][1])
        logger.info("Slowest: %s (%.0fs)", times[-1][0], times[-1][1])

    return 0 if failed == 0 else 1


def run_jobs(
    jobs: list[Job],
    devices: list[int],
    *,
    output: str,
    dry_run: bool = False,
) -> int:
    """Dispatch ``jobs`` across ``devices`` and return a process exit code.

    Args:
        jobs: List of :class:`Job` instances to execute.
        devices: GPU indices to dispatch across. With one device jobs run
            sequentially; with multiple devices each device gets a worker
            thread that pulls from a shared queue.
        output: CSV path passed as ``output=<path>`` to every invocation.
        dry_run: If ``True``, log the planned jobs and return 0 without
            running anything.

    Returns:
        ``0`` if every job succeeded, ``1`` otherwise (or if ``jobs`` is
        empty, ``0`` with a warning log).
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    total = len(jobs)
    logger.info("Running %d jobs on devices %s; output=%s, resume=true", total, devices, output)

    if total == 0:
        logger.warning("No jobs to run.")
        return 0

    if dry_run:
        for i, job in enumerate(jobs, start=1):
            gpu = devices[(i - 1) % len(devices)]
            logger.info("[%d/%d] %s -> cuda:%d", i, total, job.label, gpu)
            logger.info(
                "torchgeo-bench run %s device=cuda:%d output=%s resume=true",
                " ".join(job.overrides),
                gpu,
                output,
            )
        logger.info("Dry run complete: %d jobs across %d devices", total, len(devices))
        return 0

    job_queue: Queue[tuple[int, Job] | None] = Queue()
    for i, job in enumerate(jobs, start=1):
        job_queue.put((i, job))

    for _ in devices:
        job_queue.put(None)
    results: Queue[_JobResult] = Queue()

    start_time = time.time()
    threads: list[threading.Thread] = []
    for gpu in devices:
        t = threading.Thread(
            target=_worker,
            args=(gpu, job_queue, total, output, results),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    return summarize_results(
        [results.get_nowait() for _ in range(results.qsize())], total, elapsed, output
    )


__all__ = [
    "REPO_ROOT",
    "Job",
    "add_devices_argument",
    "default_output",
    "run_jobs",
]


if __name__ == "__main__":  # pragma: no cover
    logger.error("This module is a helper; import from a script in experiments/.")
    sys.exit(2)
