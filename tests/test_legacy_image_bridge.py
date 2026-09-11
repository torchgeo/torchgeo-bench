"""Transitional callers stop at the single legacy input boundary."""

from pathlib import Path
from unittest import mock

import pandas as pd

from torchgeo_bench.config import compose_config
from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.legacy_config import accept_legacy_config
from torchgeo_bench.main import main

from .test_main_fast import _synthetic_embeddings, _synthetic_loaders


def test_bridge_converts_old_fields_and_keeps_metadata_out_of_constructor_kwargs() -> None:
    configs = []
    capture = accept_legacy_config(lambda config, **kwargs: configs.append(config))
    capture(
        compose_config(
            [
                "model=torchgeo/scalemae_large_fmow",
                "device=cpu",
                "seed=17",
                "dataset.names=[m-eurosat,m-forestnet]",
                "eval.skip_linear=true",
                "eval.segmentation.cache_features=false",
                "eval.intrinsic_dim.estimators=[]",
                "eval.profile.enabled=true",
            ]
        )
    )
    assert len(configs) == 2
    assert all(isinstance(config, RunConfig) for config in configs)
    assert configs[0].model.kwargs["res"] != configs[1].model.kwargs["res"]
    assert configs[0].runtime.seed == 17
    assert configs[0].classification.methods == ["knn"]
    assert not configs[0].segmentation.cache_features
    assert configs[0].intrinsic_dim.estimators == []
    assert configs[0].profile.enabled
    assert (
        not {"name", "eval", "image_size", "interpolation", "dataset_overrides"}
        & configs[0].model.kwargs.keys()
    )


def test_old_main_caller_still_runs_and_persists(tmp_path: Path) -> None:
    output = tmp_path / "legacy.csv"
    config = compose_config(
        [
            "model=rcf",
            "device=cpu",
            "dataset.names=[m-eurosat]",
            "dataset.num_workers=0",
            "eval.skip_linear=true",
            "eval.bootstrap=2",
            f"output={output}",
        ]
    )
    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
    ):
        main(config)
    assert pd.read_csv(output)["method"].tolist() == ["knn5"]
