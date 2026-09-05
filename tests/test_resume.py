"""Regression tests for resume-mode config fingerprinting."""

from torchgeo_bench.config import compose_config
from torchgeo_bench.resume import _resume_config_hash


def _cfg(overrides):
    return compose_config(["model=rcf", "dataset.names=[m-eurosat]", *overrides])


def test_config_hash_ignores_profile_toggle():
    """Turning on eval.profile must not change the fingerprint.

    Profile is an additive, independently-gated follow-up pass (its own
    skip_profile resume key already tracks completion). If it changed the
    fingerprint, a follow-up `eval.profile.enabled=true resume=true` run --
    the documented way to backfill profile rows on already-evaluated models
    -- would see every existing knn/linear row as "a different config" and
    rerun (and duplicate) them instead of just adding the missing profile
    rows.
    """
    without_profile = _cfg([])
    with_profile = _cfg(["eval.profile.enabled=true", "eval.profile.cpu_throughput.enabled=true"])

    assert _resume_config_hash(without_profile) == _resume_config_hash(with_profile)


def test_config_hash_ignores_intrinsic_dim_toggle():
    """Turning on eval.intrinsic_dim must not change the fingerprint, for the same reason."""
    without_id = _cfg([])
    with_id = _cfg(["eval.intrinsic_dim.enabled=true"])

    assert _resume_config_hash(without_id) == _resume_config_hash(with_id)


def test_config_hash_changes_with_normalization():
    """Sanity check: the fingerprint must still change for settings that affect the row."""
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
