"""Unit tests for CLI entrypoints."""

import pytest

from torchgeo_bench.cli import main as cli_main


def test_run_composes_and_calls_main(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "model=rcf", "dataset.batch_size=8", "--device", "cpu"])
    assert len(received) == 1
    cfg = received[0]
    assert cfg.model.name == "rcf"
    assert cfg.dataset.batch_size == 8
    assert cfg.device == "cpu"


def test_run_flags_win_over_positional_overrides(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "device=cuda:0", "--device", "cpu"])
    assert received[0].device == "cpu"


def test_run_datasets_flag_becomes_list(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    cli_main(["run", "-d", "m-eurosat,m-so2sat"])
    assert list(received[0].dataset.names) == ["m-eurosat", "m-so2sat"]


def test_run_unknown_model_errors() -> None:
    with pytest.raises(SystemExit, match="Unknown model config"):
        cli_main(["run", "model=not_a_model"])


def test_run_unknown_key_errors() -> None:
    with pytest.raises(SystemExit, match="typo_key"):
        cli_main(["run", "typo_key=1"])


def test_run_print_config(capsys) -> None:
    cli_main(["run", "model=rcf", "--print-config"])
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
    """The flops subparser has no --datasets flag; flag translation must not assume one."""
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


def test_bad_override_is_not_a_traceback() -> None:
    with pytest.raises(SystemExit, match="bad config override"):
        cli_main(["run", "typo.key=1"])


def test_double_plus_override_sets_the_real_key():
    """`++key=value` must override, not create a literal `+key` node.

    Hydra spelled add as `+key` and add-or-override as `++key`; stripping only
    one `+` left a `+model` section and silently no-op'd the intended override,
    which the sweep scripts rely on.
    """
    from torchgeo_bench.config import compose_config

    cfg = compose_config(["model=rcf", "++model.pool=cls", "+model.gsd=1.0"])
    assert "+model" not in cfg
    assert cfg.model.pool == "cls"
    assert float(cfg.model.gsd) == 1.0


def test_model_names_are_posix_on_every_platform():
    """Config names are CLI identifiers, so they must not use OS separators.

    On Windows `str(Path)` produced `torchgeo\\scalemae_large_fmow`, which no
    documented command, config, or sweep script would match.
    """
    from torchgeo_bench.config import list_model_configs

    names = list_model_configs()
    assert "torchgeo/scalemae_large_fmow" in names
    assert not any("\\" in n for n in names)
