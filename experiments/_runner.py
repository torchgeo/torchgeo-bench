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
from queue import Empty, Queue

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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the job list without executing anything.",
    )


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

    print(f"[{idx}/{total}] START  {job.label} on cuda:{gpu}", flush=True)
    start = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - start

    if proc.returncode == 0:
        print(
            f"[{idx}/{total}] DONE   {job.label} ({elapsed:.0f}s) on cuda:{gpu}",
            flush=True,
        )
        return _JobResult(job.label, gpu, elapsed, 0)

    stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
    print(
        f"[{idx}/{total}] FAILED {job.label} ({elapsed:.0f}s) on cuda:{gpu}\n  {stderr_tail}",
        flush=True,
    )
    return _JobResult(job.label, gpu, elapsed, proc.returncode, stderr_tail)


def _worker(
    gpu: int,
    job_queue: "Queue[tuple[int, Job]]",
    total: int,
    output: str,
    results: list[_JobResult],
    lock: threading.Lock,
) -> None:
    """Pull jobs off the queue and run them on the assigned GPU until empty."""
    while True:
        try:
            idx, job = job_queue.get_nowait()
        except Empty:
            return
        try:
            result = _run_one(job, gpu, idx, total, output)
            with lock:
                results.append(result)
        finally:
            job_queue.task_done()


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
        dry_run: If ``True``, print the planned jobs and return 0 without
            running anything.

    Returns:
        ``0`` if every job succeeded, ``1`` otherwise (or if ``jobs`` is
        empty, ``0`` with a warning log).
    """
    total = len(jobs)
    print("=" * 60)
    print(f"Jobs:    {total}")
    print(f"Devices: {devices} ({len(devices)} worker{'s' if len(devices) != 1 else ''})")
    print(f"Output:  {output}")
    print("Resume:  true")
    print("=" * 60)

    if total == 0:
        logger.warning("No jobs to run.")
        return 0

    if dry_run:
        for i, job in enumerate(jobs, start=1):
            gpu = devices[(i - 1) % len(devices)]
            print(f"  [{i}/{total}] {job.label} -> cuda:{gpu}")
            print(
                f"      torchgeo-bench run {' '.join(job.overrides)} "
                f"device=cuda:{gpu} output={output} resume=true"
            )
        print(f"\n[DRY RUN] {total} job(s) across {len(devices)} device(s)")
        return 0

    job_queue: Queue[tuple[int, Job]] = Queue()
    for i, job in enumerate(jobs, start=1):
        job_queue.put((i, job))

    results: list[_JobResult] = []
    lock = threading.Lock()

    start_time = time.time()
    threads: list[threading.Thread] = []
    for gpu in devices:
        t = threading.Thread(
            target=_worker,
            args=(gpu, job_queue, total, output, results, lock),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - start_time
    passed = sum(1 for r in results if r.returncode == 0)
    failed = total - passed

    print()
    print("=" * 60)
    print("Run Complete")
    print("=" * 60)
    print(f"Total:    {total}")
    print(f"Passed:   {passed}")
    print(f"Failed:   {failed}")
    print(f"Time:     {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"Output:   {output}")

    if failed:
        print("\nFailed jobs:")
        for r in results:
            if r.returncode != 0:
                print(f"  {r.label} ({r.elapsed:.0f}s, cuda:{r.gpu})")

    if passed:
        times = sorted(
            [(r.label, r.elapsed) for r in results if r.returncode == 0],
            key=lambda x: x[1],
        )
        avg = sum(t for _, t in times) / len(times)
        print(f"\nAvg time per job: {avg:.0f}s ({avg / 60:.1f}m)")
        print(f"Fastest: {times[0][0]} ({times[0][1]:.0f}s)")
        print(f"Slowest: {times[-1][0]} ({times[-1][1]:.0f}s)")

    return 0 if failed == 0 else 1


def main_with_runner(
    parser: argparse.ArgumentParser,
    build_jobs,  # callable[[argparse.Namespace], tuple[list[Job], str]]
) -> int:
    """Convenience wrapper used by the non-custom scripts.

    Args:
        parser: Argparse parser with all script-specific args already added
            (including the standard ``--devices`` / ``--dry-run`` registered
            via :func:`add_devices_argument`).
        build_jobs: Callable taking the parsed args and returning
            ``(jobs, output_path)``.

    Returns:
        Process exit code (forward to ``sys.exit``).
    """
    args = parser.parse_args()
    jobs, output = build_jobs(args)
    return run_jobs(jobs, args.devices, output=output, dry_run=args.dry_run)


__all__ = [
    "Job",
    "add_devices_argument",
    "main_with_runner",
    "run_jobs",
    "REPO_ROOT",
]


if __name__ == "__main__":  # pragma: no cover
    print("This module is a helper; import from a script in experiments/.", file=sys.stderr)
    sys.exit(2)
