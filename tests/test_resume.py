"""Regression tests for compatible typed image fingerprints."""

from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.image_hash import _historical_payload, _resume_config_payload
from torchgeo_bench.resume import _resume_config_hash


def _cfg(**sections) -> RunConfig:
    return RunConfig.model_validate(
        {"model": {"name": "rcf"}, "datasets": ["m-eurosat"], **sections}
    )


def test_config_hash_ignores_profile_toggle():
    assert _resume_config_hash(_cfg()) == _resume_config_hash(
        _cfg(profile={"enabled": True, "cpu_throughput": {"enabled": True}})
    )


def test_config_hash_ignores_intrinsic_dim_toggle():
    assert _resume_config_hash(_cfg()) == _resume_config_hash(_cfg(intrinsic_dim={"enabled": True}))


def test_config_hash_changes_with_normalization():
    assert _resume_config_hash(_cfg(input={"normalization": "dataset"})) != _resume_config_hash(
        _cfg(input={"normalization": "minmax"})
    )


def test_removed_plot_defaults_keep_existing_resume_keys():
    payload = _resume_config_payload(_cfg())
    assert payload == _historical_payload(_cfg(), "legacy")
    segmentation = payload["eval"]["segmentation"]
    assert segmentation["save_viz"] is False
    assert segmentation["viz_dir"] == "viz"
    assert segmentation["n_viz_samples"] == 8
    assert not {"save_viz", "viz_dir", "n_viz_samples"} & _cfg().segmentation.model_fields_set


def test_dataset_selection_does_not_change_hash():
    assert _resume_config_hash(_cfg()) == _resume_config_hash(_cfg(datasets=["m-forestnet"]))
