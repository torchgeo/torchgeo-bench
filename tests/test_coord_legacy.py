"""Coverage for the temporary legacy coordinate caller boundary."""

from pathlib import Path
from typing import Any

import pytest

from torchgeo_bench.config import compose_config
from torchgeo_bench.coordbench.config import CoordConfig
from torchgeo_bench.coordbench.legacy import accepts_legacy_config, legacy_coord_config


def test_legacy_boundary_translates_coordinate_fields(tmp_path: Path) -> None:
    config = compose_config(
        [
            "mode=coord",
            "model=sincos",
            "device=cpu",
            "seed=42",
            "resume=true",
            "coord.names=country,worldclim",
            "coord.methods=[linear]",
            "coord.knn_device=null",
            f"coord.output={tmp_path / 'results.csv'}",
            "+model.batch_size=32",
        ]
    )
    typed = legacy_coord_config(config)
    assert typed.model.name == "sincos"
    assert typed.model.target == "torchgeo_bench.coordbench.models.SinCosLocationEncoder"
    assert typed.model.kwargs == {"batch_size": 32}
    assert typed.datasets == ["country", "worldclim"]
    assert typed.evaluation.methods == ["linear"]
    assert typed.evaluation.knn_device == "cpu"
    assert typed.runtime.seed == 42
    assert typed.runtime.device == "cpu"
    assert typed.output.resume
    assert typed.output.file == str(tmp_path / "results.csv")
    assert "_target_" in config.model
    assert config.coord.output == str(tmp_path / "results.csv")


def test_legacy_decorator_preserves_typed_boundary() -> None:
    configs = []

    @accepts_legacy_config
    def runtime(config: CoordConfig) -> None:
        configs.append(config)

    config: Any = compose_config(["mode=coord", "model=sincos", "device=cpu"])
    runtime(config)
    assert len(configs) == 1
    assert isinstance(configs[0], CoordConfig)
    runtime(configs[0])
    assert configs[0] is configs[1]


def test_legacy_adapter_rejects_other_objects() -> None:
    with pytest.raises(TypeError, match="CoordConfig"):
        legacy_coord_config({"model": {"name": "sincos"}})


def test_legacy_main_coord_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from torchgeo_bench.main import main

    monkeypatch.setattr("torchgeo_bench.coordbench.run.load_benchmarks", lambda names: [])
    config = compose_config(
        [
            "mode=coord",
            "model=sincos",
            "device=cpu",
            f"coord.output={tmp_path / 'coord.csv'}",
        ]
    )
    main(config)
    assert not (tmp_path / "coord.csv").exists()
