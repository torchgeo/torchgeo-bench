"""Regression tests for standalone segmentation sweep runners."""

import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).parents[1]


def _load_script(filename: str) -> ModuleType:
    """Load a script module without requiring ``scripts`` to be a package.

    The sweep and study scripts both import ``_seg_sweep_common`` as a
    top-level module, so ``scripts/`` must be on ``sys.path`` before exec.
    """
    scripts_dir = str(ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    path = ROOT / "scripts" / filename
    module_name = f"test_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_representative_sweep_passes_seed_and_rejects_unknown_metadata(tmp_path: Path) -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")
    config = sweep.SweepConfig(
        root=ROOT,
        cli=tmp_path / "torchgeo-bench",
        output=tmp_path / "results.csv",
        state_dir=tmp_path / "state",
        gpus=[0],
        image_size=224,
        num_workers=2,
        max_attempts=2,
        seed=17,
    )
    runner = sweep.SweepRunner(config)

    assert "seed=17" in runner._command(runner.jobs[0], gpu=0, attempt=1)
    assert sweep.sweep_metadata(ROOT, image_size=224, seed=17)["seed"] == 17

    config.output.write_text("dataset\n")
    runner.metadata_path.write_text('{"schema_version": 1}\n')
    with pytest.raises(RuntimeError, match="incompatible sweep configuration"):
        runner._validate_metadata()


def test_protocol_study_passes_configured_seed(tmp_path: Path) -> None:
    study = _load_script("run_segmentation_protocol_study.py")
    config = study.StudyConfig(
        root=ROOT,
        cli=tmp_path / "torchgeo-bench",
        raw_dir=tmp_path / "raw",
        state_dir=tmp_path / "state",
        combined_output=tmp_path / "combined.csv",
        gpus=[0],
        num_workers=2,
        max_attempts=2,
        seed=23,
    )
    runner = study.StudyRunner(config)

    assert "seed=23" in runner._command(runner.jobs[0], gpu=0, attempt=1)
    assert study.study_metadata(ROOT, seed=23)["seed"] == 23


def test_queue_dry_run_logs_without_launching_jobs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    runner = _load_script("../experiments/_runner.py")

    def fail_if_called(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A dry run must not launch a subprocess")

    monkeypatch.setattr(runner.subprocess, "run", fail_if_called)
    caplog.set_level(logging.INFO, logger=runner.__name__)
    jobs = [runner.Job("first", ["model=rcf"]), runner.Job("second", ["model=timm/resnet18"])]

    assert runner.run_jobs(jobs, [0, 2], output="results.csv", dry_run=True) == 0
    assert (
        "torchgeo-bench run model=rcf device=cuda:0 output=results.csv resume=true" in caplog.text
    )
    assert "torchgeo-bench run model=timm/resnet18 device=cuda:2" in caplog.text


@pytest.mark.parametrize("returncode", [0, 1])
def test_queue_reports_job_result_at_the_expected_log_level(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, returncode: int
) -> None:
    runner = _load_script("../experiments/_runner.py")

    def run_command(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(command, returncode, "", "failed to load checkpoint")

    monkeypatch.setattr(runner.subprocess, "run", run_command)
    caplog.set_level(logging.INFO, logger=runner.__name__)
    assert (
        runner.run_jobs([runner.Job("rcf", ["model=rcf"])], [0], output="results.csv") == returncode
    )
    records = [record for record in caplog.records if record.name == runner.__name__]
    assert any("Run complete" in record.message for record in records)
    if returncode:
        assert any(
            record.levelno == logging.ERROR and "failed to load checkpoint" in record.message
            for record in records
        )
    else:
        assert any(
            record.levelno == logging.INFO and "DONE rcf" in record.message for record in records
        )
        assert all(record.levelno < logging.ERROR for record in records)


def test_queue_logging_uses_stderr_by_default() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from experiments._runner import Job, run_jobs; "
            "run_jobs([Job('rcf', ['model=rcf'])], [0], output='results.csv', dry_run=True)",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == ""
    assert "Dry run complete" in result.stderr
