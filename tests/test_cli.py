"""Unit tests for CLI entrypoints."""

import pytest

from torchgeo_bench.cli import main as cli_main


def test_run_composes_and_calls_main(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "-m", "rcf", "--batch-size", "8", "--device", "cpu"])
    assert len(received) == 1
    cfg = received[0]
    assert cfg.model["name"] == "rcf"
    assert cfg.dataset.batch_size == 8
    assert cfg.device == "cpu"


def test_run_flags_only_the_last_device_wins(monkeypatch) -> None:
    """argparse itself enforces "last flag wins" for a repeated option."""
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "--device", "cuda:0", "--device", "cpu"])
    assert received[0].device == "cpu"


def test_run_datasets_flag_becomes_list(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "-d", "m-eurosat,m-so2sat"])
    assert list(received[0].dataset.names) == ["m-eurosat", "m-so2sat"]


def test_run_exposes_common_sweep_flags_without_a_config_yaml(monkeypatch) -> None:
    """These are the values existing segmentation sweep scripts vary per job;
    exposing them as flags means a script never has to write a per-job YAML."""
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(
        [
            "run",
            "-m",
            "timm/resnet50",
            "-d",
            "burn_scars",
            "--num-workers",
            "8",
            "--time-steps",
            "4",
            "--no-merge-val",
            "--knn-device",
            "cpu",
            "--seg-head",
            "dpt",
            "--seg-epochs",
            "20",
            "--seg-lr",
            "0.0005",
            "--seg-scheduler",
            "none",
            "--seg-batch-size",
            "32",
            "--no-seg-cache",
            "--seg-cache-dtype",
            "float32",
        ]
    )
    cfg = received[0]
    assert cfg.dataset.num_workers == 8
    assert cfg.dataset.time_steps == 4
    assert cfg.eval.merge_val is False
    assert cfg.eval.knn_device == "cpu"
    seg = cfg.eval.segmentation
    assert seg.head_type == "dpt"
    assert seg.epochs == 20
    assert seg.lr == 0.0005
    assert seg.lr_scheduler == "none"
    assert seg.batch_size == 32
    assert seg.cache_features is False
    assert seg.cache_dtype == "float32"


def test_run_sweep_flags_are_omitted_by_default(monkeypatch) -> None:
    """Omitting these flags must leave the settings defaults untouched, not
    force e.g. merge_val=true / cache_features=true onto the wire."""
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "-m", "rcf"])
    cfg = received[0]
    assert cfg.dataset.num_workers == 4
    assert cfg.dataset.time_steps is None
    assert cfg.eval.merge_val is True
    assert cfg.eval.knn_device is None
    assert cfg.eval.segmentation.head_type == "fpn"
    assert cfg.eval.segmentation.cache_features is True


def test_run_exposes_model_level_ablation_flags(monkeypatch) -> None:
    """These are the values existing ablation scripts vary per model job via
    dotlist overrides against the model config's own dict."""
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(
        [
            "run",
            "-m",
            "timm/vit/vit_base_patch16_224",
            "--interpolation",
            "bicubic",
            "--use-cls-token",
            "--model-input-normalization",
            "imagenet",
            "--model-name",
            "vit_base_cls_imagenet",
        ]
    )
    cfg = received[0]
    assert cfg.dataset.interpolation == "bicubic"
    assert cfg.model["use_cls_token"] is True
    assert cfg.model["input_normalization"] == "imagenet"
    assert cfg.model["name"] == "vit_base_cls_imagenet"
    # Untouched model-config fields survive the override.
    assert cfg.model["model_name"] == "vit_base_patch16_224"
    assert cfg.model["_target_"] == "torchgeo_bench.models.TimmPatchBenchModel"


def test_run_model_level_ablation_flags_omitted_by_default(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "-m", "timm/vit/vit_base_patch16_224"])
    cfg = received[0]
    assert cfg.dataset.interpolation == "bilinear"
    assert cfg.model["use_cls_token"] is False  # the model config's own default
    assert "input_normalization" not in cfg.model
    assert cfg.model["name"] == "vit_base_patch16_224"


def test_run_no_use_cls_token_explicitly_sets_false(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "-m", "timm/vit/vit_base_patch16_224", "--no-use-cls-token"])
    assert received[0].model["use_cls_token"] is False


def test_run_unknown_model_errors() -> None:
    with pytest.raises(SystemExit, match="Unknown model config"):
        cli_main(["run", "-m", "not_a_model"])


def test_run_config_yaml_is_overridden_by_flags(monkeypatch, tmp_path) -> None:
    """Precedence: Python defaults -> --config YAML -> explicit CLI flags."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("device: cuda:1\ndataset:\n  batch_size: 32\n")
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)

    cli_main(["run", "-m", "rcf", "--config", str(config_path)])
    assert received[0].device == "cuda:1"
    assert received[0].dataset.batch_size == 32

    received.clear()
    cli_main(["run", "-m", "rcf", "--config", str(config_path), "--device", "cpu"])
    assert received[0].device == "cpu"  # flag wins over --config
    assert received[0].dataset.batch_size == 32  # --config still wins over the Python default


def test_run_config_yaml_rejects_unknown_setting(tmp_path) -> None:
    """A typo in --config is a startup error, not a silently-ignored setting."""
    config_path = tmp_path / "settings.yaml"
    config_path.write_text("daatset:\n  batch_size: 32\n")
    with pytest.raises(SystemExit, match="Unknown setting 'daatset'"):
        cli_main(["run", "-m", "rcf", "--config", str(config_path)])


def test_run_print_config(capsys) -> None:
    cli_main(["run", "-m", "rcf", "--print-config"])
    out = capsys.readouterr().out
    assert "name: rcf" in out
    assert "names: all" in out


def test_run_list_models(capsys) -> None:
    cli_main(["run", "--list-models"])
    out = capsys.readouterr().out
    assert "rcf" in out
    assert "torchgeo/scalemae_large_fmow" in out


def test_run_list_datasets(capsys) -> None:
    cli_main(["run", "--list-datasets"])
    out = capsys.readouterr().out
    assert "m-eurosat" in out


def test_run_model_help(capsys) -> None:
    cli_main(["run", "--model-help", "rcf"])
    out = capsys.readouterr().out
    assert "torchgeo_bench.models.RCFBench" in out


def test_run_model_help_unknown_model_errors() -> None:
    with pytest.raises(SystemExit, match="error: Unknown model config"):
        cli_main(["run", "--model-help", "not-a-real-model"])


def test_run_model_help_rejects_path_traversal() -> None:
    with pytest.raises(SystemExit, match="error: Unknown model config"):
        cli_main(["run", "--model-help", "../flops_config"])


def test_profile_enables_profile_pass(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["profile", "-m", "rcf", "--device", "cpu"])
    assert received[0].eval.profile.enabled is True
    assert received[0].eval.intrinsic_dim.enabled is False


def test_intrinsic_dim_enables_intrinsic_dim_pass(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["intrinsic-dim", "-m", "rcf", "--device", "cpu"])
    assert received[0].eval.intrinsic_dim.enabled is True
    assert received[0].eval.profile.enabled is False


def test_profile_shares_list_models_with_run(capsys) -> None:
    cli_main(["profile", "--list-models"])
    out = capsys.readouterr().out
    assert "rcf" in out


def test_coord_forces_coord_mode_and_requires_model(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["coord", "-m", "sincos", "--split", "both", "--names", "pdfm"])
    cfg = received[0]
    assert cfg.mode == "coord"
    assert cfg.model["name"] == "sincos"
    assert cfg.coord.split == "both"
    assert list(cfg.coord.names) == ["pdfm"]


def test_coord_without_model_errors_cleanly() -> None:
    with pytest.raises(SystemExit, match="No model selected"):
        cli_main(["coord"])


def test_download_invalid_target(capsys) -> None:
    with pytest.raises(SystemExit):
        cli_main(["download", "bogus"])
    assert "invalid choice" in capsys.readouterr().err


def test_download_geobench_v1(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def _fake_download(path, datasets=None) -> None:
        calls.append((str(path), datasets))

    monkeypatch.setattr("torchgeo_bench.download.download_geobench_v1", _fake_download)
    cli_main(["download", "geobench_v1"])
    assert calls == [("data", None)]


def test_download_geobench_v2_with_datasets(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def _fake_download(path, datasets=None) -> None:
        calls.append((str(path), datasets))

    monkeypatch.setattr("torchgeo_bench.download.download_geobench_v2", _fake_download)
    cli_main(["download", "geobench_v2", "--datasets", "burn_scars,benv2"])
    assert calls == [("data", ["burn_scars", "benv2"])]


def test_download_eurosat(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "torchgeo_bench.download.download_eurosat", lambda path: calls.append(str(path))
    )
    cli_main(["download", "eurosat"])
    assert calls == ["data"]


def test_download_rejects_empty_dataset_list() -> None:
    with pytest.raises(SystemExit, match="at least one dataset name"):
        cli_main(["download", "geobench_v1", "--datasets", ","])


def test_download_resisc45(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "torchgeo_bench.download.download_resisc45", lambda path: calls.append(str(path))
    )
    cli_main(["download", "resisc45"])
    assert calls == ["data"]


def test_download_rejects_datasets_for_resisc45() -> None:
    with pytest.raises(SystemExit):
        cli_main(["download", "resisc45", "--datasets", "resisc45"])


def test_download_rejects_datasets_for_eurosat() -> None:
    with pytest.raises(SystemExit, match="only supported for GeoBench"):
        cli_main(["download", "eurosat", "--datasets", "m-eurosat"])


def test_flops_without_model_errors_cleanly() -> None:
    with pytest.raises(SystemExit, match="No model selected"):
        cli_main(["flops"])


def test_flops_composes_flops_config(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.flops_pipeline.main", received.append)
    cli_main(["flops", "-m", "rcf", "--device", "cpu"])
    assert received[0].probe_num_classes == 10
    assert received[0].device == "cpu"


def test_unknown_model_suggests_close_names() -> None:
    with pytest.raises(SystemExit, match="timm/resnet50"):
        cli_main(["run", "-m", "resnet50"])


def test_run_rejects_unrecognized_positional_arguments() -> None:
    """The old `key=value` positional-override grammar is gone."""
    with pytest.raises(SystemExit):
        cli_main(["run", "dataset.batch_size=8"])


def test_nested_mapping_overrides_a_new_model_key():
    """CLI/--config overrides build a nested mapping directly; a model config's
    plain dict has no fixed schema, so a new key is accepted (unlike a
    settings dataclass field, which rejects unknown keys)."""
    from torchgeo_bench.config import compose_config

    cfg = compose_config({"model": {"pool": "cls", "gsd": 1.0}}, model="rcf")
    assert cfg.model["pool"] == "cls"
    assert float(cfg.model["gsd"]) == 1.0


def test_model_names_are_posix_on_every_platform():
    """Config names are CLI identifiers, so they must not use OS separators.

    On Windows `str(Path)` produced `torchgeo\\scalemae_large_fmow`, which no
    documented command, config, or sweep script would match.
    """
    from torchgeo_bench.config import list_model_configs

    names = list_model_configs()
    assert "torchgeo/scalemae_large_fmow" in names
    assert not any("\\" in n for n in names)


def test_all_model_configs_resolve_to_importable_targets() -> None:
    """Every shipped model config must resolve to JSON-native importable settings."""
    import importlib
    import json

    from torchgeo_bench.config import compose_config, list_model_configs
    from torchgeo_bench.settings import to_dict

    for name in list_model_configs():
        cfg = compose_config(model=name, default_model=None)
        model = to_dict(cfg.model)
        json.dumps(model)  # strict; raises if any value isn't JSON-native
        assert isinstance(model, dict)
        module_name, _, class_name = str(model["_target_"]).rpartition(".")
        assert hasattr(importlib.import_module(module_name), class_name)
