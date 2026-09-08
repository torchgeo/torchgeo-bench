"""Regression tests for resume-mode config fingerprinting."""

from torchgeo_bench.config import compose_config
from torchgeo_bench.resume import _resume_config_hash


def _cfg(overrides):
    return compose_config(["model=rcf", "dataset.names=[m-eurosat]", *overrides])


def test_config_hash_ignores_profile_toggle():
    """Adding profile rows with ``eval.profile.enabled=true resume=true`` must not rerun completed probes."""
    without_profile = _cfg([])
    with_profile = _cfg(["eval.profile.enabled=true", "eval.profile.cpu_throughput.enabled=true"])

    assert _resume_config_hash(without_profile) == _resume_config_hash(with_profile)


def test_config_hash_ignores_intrinsic_dim_toggle():
    """Adding intrinsic-dimension rows must not rerun completed probes."""
    without_id = _cfg([])
    with_id = _cfg(["eval.intrinsic_dim.enabled=true"])

    assert _resume_config_hash(without_id) == _resume_config_hash(with_id)


def test_config_hash_changes_with_normalization():
    """Normalization changes the evaluated inputs, so it must change the resume key."""
    zscore = _cfg(["dataset.normalization=bandspec_zscore"])
    minmax = _cfg(["dataset.normalization=minmax"])

    assert _resume_config_hash(zscore) != _resume_config_hash(minmax)


def test_removed_plot_defaults_keep_existing_resume_keys() -> None:
    current = _cfg([])
    previous = _cfg(
        [
            "+eval.segmentation.save_viz=false",
            "+eval.segmentation.viz_dir=viz",
            "+eval.segmentation.n_viz_samples=8",
        ]
    )
    assert _resume_config_hash(current) == _resume_config_hash(previous)
    assert not {"save_viz", "viz_dir", "n_viz_samples"} & set(current.eval.segmentation)
