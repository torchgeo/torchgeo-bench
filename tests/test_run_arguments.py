"""Tests for standalone run argument registration."""

import argparse
from pathlib import Path

import pytest

from torchgeo_bench.commands.run_arguments import add_run_arguments


def test_omitted_flags_do_not_set_config_defaults() -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    assert vars(parser.parse_args([])) == {}


def test_explicit_values_preserve_types_and_dataset_order() -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    args = parser.parse_args(
        [
            "--config",
            "run.yaml",
            "-m",
            "rcf",
            "-d",
            "m-eurosat",
            "--dataset",
            "burn_scars",
            "--image-size",
            "none",
            "--workers",
            "0",
            "--no-resume",
            "--methods",
            "linear",
            "--dry-run",
        ]
    )
    assert vars(args) == {
        "config": Path("run.yaml"),
        "model": "rcf",
        "datasets": ["m-eurosat", "burn_scars"],
        "image_size": None,
        "workers": 0,
        "resume": False,
        "methods": ["linear"],
        "dry_run": True,
    }


@pytest.mark.parametrize("size", ["0", "-1", "bad"])
def test_invalid_image_sizes_are_parser_errors(size: str) -> None:
    parser = argparse.ArgumentParser()
    add_run_arguments(parser)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--image-size", size])
    assert error.value.code == 2
