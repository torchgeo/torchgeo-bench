"""Fast offline tests for the intrinsic-dimension branch in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import pandas as pd

from torchgeo_bench.main import main

from .test_main_fast import _compose_cfg, _resume_row, _synthetic_embeddings, _synthetic_loaders


def test_intrinsic_dim_rows_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=[
            "eval.skip_linear=true",
            "eval.intrinsic_dim.enabled=true",
            "eval.intrinsic_dim.estimators=[twonn,mle]",
            "eval.intrinsic_dim.splits=[train]",
            "eval.intrinsic_dim.max_samples=100",
        ],
    )

    def _mock_compute(*args, **kwargs):
        est = str(kwargs["estimators"][0])
        values = {"twonn": 5.0, "mle": 4.8}
        return {est: values[est]}

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("torchgeo_bench.main.evaluate_knn", return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6)),
        mock.patch("torchgeo_bench.main.compute_intrinsic_dim", side_effect=_mock_compute),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    id_df = df[df["method"] == "intrinsic_dim"]
    assert not id_df.empty
    assert "id_twonn_train" in id_df["metric_name"].values
    assert "id_mle_train" in id_df["metric_name"].values


def test_intrinsic_dim_resume_per_estimator(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=[
            "resume=true",
            "eval.skip_linear=true",
            "eval.intrinsic_dim.enabled=true",
            "eval.intrinsic_dim.estimators=[twonn,mle]",
            "eval.intrinsic_dim.splits=[train]",
            "eval.intrinsic_dim.max_samples=100",
        ],
    )

    seed_rows = [
        _resume_row(cfg, method="knn5", metric_name="accuracy"),
        _resume_row(cfg, method="intrinsic_dim", metric_name="id_twonn_train"),
    ]
    pd.DataFrame(seed_rows).to_csv(out, index=False)

    def _mock_compute(*args, **kwargs):
        est = str(kwargs["estimators"][0])
        values = {"twonn": 5.0, "mle": 4.8}
        return {est: values[est]}

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("torchgeo_bench.main.evaluate_knn", return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6)),
        mock.patch("torchgeo_bench.main.compute_intrinsic_dim", side_effect=_mock_compute),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    id_df = df[df["method"] == "intrinsic_dim"]
    assert int((id_df["metric_name"] == "id_twonn_train").sum()) == 1
    assert int((id_df["metric_name"] == "id_mle_train").sum()) == 1
