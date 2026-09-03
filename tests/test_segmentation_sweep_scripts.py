"""Regression tests for standalone segmentation sweep runners."""

import importlib.util
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
    assert spec is not None and spec.loader is not None
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


def test_representative_sweep_filters_and_deduplicates_band_modes() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")
    model = sweep.MODELS[0]

    jobs = sweep.build_jobs(
        models=[model],
        datasets=["caffe", "spacenet2", "spacenet7"],
        heads=["fpn"],
        bands=["rgb", "all"],
    )

    assert [(job.dataset, job.bands) for job in jobs] == [
        ("caffe", "rgb"),
        ("spacenet2", "rgb"),
        ("spacenet2", "all"),
        ("spacenet7", "rgb"),
    ]


def test_representative_sweep_enforces_model_input_contracts() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")
    for config in [
        "torchgeo/scalemae_large_fmow",
        "torchgeo/scalemae_large_fmow_cls",
    ]:
        scalemae = sweep._select_models(config)[0]
        jobs = sweep.build_jobs(
            models=[scalemae],
            datasets=["dynamic_earthnet", "spacenet2", "spacenet7"],
            heads=["fpn"],
            bands=["rgb", "all"],
        )

        assert [(job.dataset, job.bands) for job in jobs] == [
            ("spacenet2", "rgb"),
            ("spacenet7", "rgb"),
        ]


def test_representative_sweep_selects_models_by_config_or_name() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")

    selected = sweep._select_models("timm/resnet50,convnext_tiny")

    assert [model.name for model in selected] == ["resnet50", "convnext_tiny"]


def test_representative_sweep_selects_arbitrary_segmentation_config() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")

    selected = sweep._select_models("resnet18")

    assert selected == [sweep.Model("timm/resnet18", "resnet18", 8, 8)]


def test_representative_sweep_rejects_model_without_segmentation_layers() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")

    with pytest.raises(ValueError, match="segmentation-incompatible"):
        sweep._select_models("rcf")


def test_representative_sweep_selects_all_configured_segmentation_models() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")

    selected = sweep._select_models("configured")

    assert len(selected) >= 80
    assert len({model.name for model in selected}) == len(selected)
    assert "resnet18" in {model.name for model in selected}
    assert "rcf" not in {model.name for model in selected}


def test_representative_sweep_selects_all_band_mode_literally() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")

    selected = sweep._parse_choices("all", sweep.BANDS, "band modes", expand_all=False)

    assert selected == ["all"]


def test_representative_sweep_deduplicates_repeated_choices() -> None:
    sweep = _load_script("run_segmentation_representative_sweep.py")

    selected = sweep._parse_choices("fpn,linear,fpn", sweep.HEADS, "heads")

    assert selected == ["fpn", "linear"]


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
