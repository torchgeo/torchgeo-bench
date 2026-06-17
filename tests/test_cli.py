"""Unit tests for CLI entrypoints."""

from typer.testing import CliRunner

from torchgeo_bench.cli import app

runner = CliRunner()


def test_run_forwards_to_hydra(monkeypatch) -> None:
    calls: list[tuple] = []

    def _fake_hydra_main() -> None:
        calls.append(())

    monkeypatch.setattr("torchgeo_bench.main.main", _fake_hydra_main)
    result = runner.invoke(app, ["run", "model=rcf"])
    assert result.exit_code == 0
    assert len(calls) == 1


def test_download_invalid_target() -> None:
    result = runner.invoke(app, ["download", "bogus"])
    assert result.exit_code == 1
    assert "Unknown target" in result.stderr


def test_download_geobench_v1(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_download(path) -> None:
        calls.append(str(path))

    monkeypatch.setattr("torchgeo_bench.download.download_geobench_v1", _fake_download)
    result = runner.invoke(app, ["download", "geobench_v1"])
    assert result.exit_code == 0
    assert len(calls) == 1


def test_download_geobench_v2_with_datasets(monkeypatch) -> None:
    calls: list[tuple[str, list[str] | None]] = []

    def _fake_download(path, datasets=None) -> None:
        calls.append((str(path), datasets))

    monkeypatch.setattr("torchgeo_bench.download.download_geobench_v2", _fake_download)
    result = runner.invoke(
        app,
        ["download", "geobench_v2", "--datasets", "burn_scars,benv2"],
    )
    assert result.exit_code == 0
    assert calls == [("data", ["burn_scars", "benv2"])]


def test_download_eurosat(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_download(path) -> None:
        calls.append(str(path))

    monkeypatch.setattr("torchgeo_bench.download.download_eurosat", _fake_download)
    result = runner.invoke(app, ["download", "eurosat"])
    assert result.exit_code == 0
    assert len(calls) == 1


def test_nf_forwards_to_hydra(monkeypatch) -> None:
    calls: list[tuple] = []

    def _fake_nf_main() -> None:
        calls.append(())

    monkeypatch.setattr("torchgeo_bench.nf_pipeline.main", _fake_nf_main)
    result = runner.invoke(app, ["nf", "model=resnet50"])
    assert result.exit_code == 0
    assert len(calls) == 1


def test_sample_size_subcommand_help_exits_cleanly() -> None:
    result = runner.invoke(app, ["sample-size", "--help"])
    assert result.exit_code == 0
    assert "sample" in result.output.lower()
