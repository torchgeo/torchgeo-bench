"""Consistent nonblank output paths across image, FLOPs, and coordinate settings."""

import pytest
from pydantic import ValidationError

from torchgeo_bench.config.flops import FlopsOutputConfig
from torchgeo_bench.config.run import OutputConfig
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
