"""Consistent nonblank output paths across image, FLOPs, and coordinate settings."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from torchgeo_bench.cli import main
from torchgeo_bench.config.flops import FlopsOutputConfig
from torchgeo_bench.config.run import OutputConfig, RunConfig
from torchgeo_bench.config.schema import StrictModel
from torchgeo_bench.coordbench.config import CoordOutputConfig


@pytest.mark.parametrize(
    ("config_class", "field", "default"),
    [
        (OutputConfig, "directory", "results/models"),
        (OutputConfig, "file", None),
        (OutputConfig, "profile_directory", "results/profiles"),
        (OutputConfig, "intrinsic_dim_directory", "results/intrinsic_dim"),
        (FlopsOutputConfig, "file", "results/compute_cost.csv"),
        (CoordOutputConfig, "file", "results/coordbench_results.csv"),
    ],
)
class TestOutputPaths:
    def test_default_is_unchanged(
        self, config_class: type[StrictModel], field: str, default: str | None
    ) -> None:
        assert config_class().model_dump()[field] == default

    @pytest.mark.parametrize("value", ["", " ", "\t\n", 1, False, 1.5])
    def test_blank_or_non_string_path_is_rejected(
        self, config_class: type[StrictModel], field: str, default: str | None, value: object
    ) -> None:
        with pytest.raises(ValidationError, match=field) as error:
            config_class.model_validate({field: value})
        assert error.value.errors()[0]["loc"] == (field,)

    @pytest.mark.parametrize(
        "value", ["results.csv", " path with spaces.csv ", "missing/tree/out.csv"]
    )
    def test_valid_path_is_not_rewritten(
        self, config_class: type[StrictModel], field: str, default: str | None, value: str
    ) -> None:
        assert config_class.model_validate({field: value}).model_dump()[field] == value

    def test_only_optional_image_file_accepts_null(
        self, config_class: type[StrictModel], field: str, default: str | None
    ) -> None:
        if config_class is OutputConfig and field == "file":
            assert config_class.model_validate({field: None}).model_dump()[field] is None
        else:
            with pytest.raises(ValidationError, match=field):
                config_class.model_validate({field: None})


@pytest.mark.parametrize("yaml_output", [False, True])
@pytest.mark.parametrize(
    ("flag", "field"),
    [
        ("", ""),
        ("--results-dir", "directory"),
        ("--profile-dir", "profile_directory"),
        ("--intrinsic-dim-dir", "intrinsic_dim_directory"),
    ],
)
def test_run_output_flags_preserve_other_settings_and_round_trip(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    field: str,
    *,
    yaml_output: bool,
) -> None:
    expected = OutputConfig()
    if yaml_output:
        expected = OutputConfig(
            directory="yaml/models",
            profile_directory="yaml/profiles",
            intrinsic_dim_directory="yaml/intrinsic_dim",
            file="yaml/all.csv",
            resume=True,
        )
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "rcf"},
                "datasets": ["m-eurosat"],
                "output": expected.model_dump_yaml(),
            }
        ),
        encoding="utf-8",
    )
    flags = []
    if flag:
        flags = [flag, "custom measurements"]
        expected = expected.model_copy(update={field: "custom measurements"})
    main(["run", "--config", str(config_path), *flags, "--dry-run"])
    output = capsys.readouterr().out
    assert RunConfig.model_validate(yaml.safe_load(output)).output == expected

    config_path.write_text(output, encoding="utf-8")
    main(["run", "--config", str(config_path), "--dry-run"])
    assert yaml.safe_load(capsys.readouterr().out) == yaml.safe_load(output)
