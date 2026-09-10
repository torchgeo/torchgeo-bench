"""Regression tests for experiment command construction."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).parents[1]


def _load_experiment(filename: str) -> ModuleType:
    experiments = ROOT / "experiments"
    if str(experiments) not in sys.path:
        sys.path.insert(0, str(experiments))
    path = experiments / filename
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_experiment_jobs_use_explicit_cli_flags() -> None:
    for filename in [
        "run_cls_token_experiment.py",
        "run_main_experiments.py",
        "run_resize_and_normalization_experiment.py",
    ]:
        module = _load_experiment(filename)
        for job in module.build_jobs():
            assert job.args
            assert all("=" not in argument for argument in job.args)
            assert job.args[0].startswith("--")
