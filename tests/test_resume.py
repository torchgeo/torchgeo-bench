"""Regression tests for resume-mode config fingerprinting."""

import pytest

from torchgeo_bench.config import compose_config
from torchgeo_bench.resume import _resume_config_hash


def _cfg(overrides: dict | None = None):
    return compose_config(overrides or {}, model="rcf")


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
    without_profile = _cfg()
    with_profile = _cfg(
        {"eval": {"profile": {"enabled": True, "cpu_throughput": {"enabled": True}}}}
    )

    assert _resume_config_hash(without_profile) == _resume_config_hash(with_profile)


def test_config_hash_ignores_intrinsic_dim_toggle():
    """Turning on eval.intrinsic_dim must not change the fingerprint, for the same reason."""
    without_id = _cfg()
    with_id = _cfg({"eval": {"intrinsic_dim": {"enabled": True}}})

    assert _resume_config_hash(without_id) == _resume_config_hash(with_id)


def test_config_hash_changes_with_normalization():
    """Sanity check: the fingerprint must still change for settings that affect the row."""
    zscore = _cfg({"dataset": {"normalization": "bandspec_zscore"}})
    minmax = _cfg({"dataset": {"normalization": "minmax"}})

    assert _resume_config_hash(zscore) != _resume_config_hash(minmax)


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("timm/convnext_base", "bdef71b44fb8fb29"),
        ("timm/convnext_large", "d8ec53e70484ac7d"),
        ("timm/convnext_small", "c73ed0c79cadee82"),
    ],
)
def test_current_generation_config_hashes_remain_stable(model: str, expected: str) -> None:
    """Configuration cleanup must not invalidate current committed result rows.

    ``device=cuda:0`` pins this test to the value the committed rows were
    actually hashed with. ``compose_config``'s own default is now ``"auto"``
    (resolved to a concrete device only inside ``main()``, depending on
    whatever machine actually runs the benchmark) -- hashing that literal
    default here would make this hardware-independent regression check flaky
    across CI runners with and without a GPU.
    """
    cfg = compose_config({"device": "cuda:0"}, model=model)

    assert _resume_config_hash(cfg) == expected
