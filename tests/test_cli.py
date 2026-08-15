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
    with pytest.raises(Exception, match="typo_key"):
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


def test_download_rejects_datasets_for_eurosat() -> None:
    with pytest.raises(SystemExit, match="only supported for GeoBench"):
        cli_main(["download", "eurosat", "--datasets", "m-eurosat"])
