"""Fast offline tests for the profile branch in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import pandas as pd

from torchgeo_bench.main import _profile_metric_names, main

from .test_main_fast import _compose_cfg, _resume_row, _synthetic_embeddings, _synthetic_loaders


def test_profile_rows_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=[
            "eval.skip_linear=true",
            "eval.profile.enabled=true",
            "eval.profile.n_warmup=1",
            "eval.profile.n_measure=1",
        ],
    )
    metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("torchgeo_bench.main.evaluate_knn", return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6)),
        mock.patch("torchgeo_bench.main.measure_profile", return_value=metrics),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    profile_df = df[df["method"] == "profile"]
    assert not profile_df.empty
    for name in metrics:
        assert name in profile_df["metric_name"].values


def test_profile_resume_partial_does_not_skip(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=[
            "resume=true",
            "eval.skip_linear=true",
            "eval.profile.enabled=true",
            "eval.profile.n_warmup=1",
            "eval.profile.n_measure=1",
        ],
    )
    metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    seed_rows = [
        _resume_row(cfg, method="knn5", metric_name="accuracy"),
        _resume_row(cfg, method="profile", metric_name="params_m"),
    ]
    pd.DataFrame(seed_rows).to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("torchgeo_bench.main.evaluate_knn", return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6)),
        mock.patch("torchgeo_bench.main.measure_profile", return_value=metrics),
    ):
        main.__wrapped__(cfg)

    df = pd.read_csv(out)
    profile_df = df[df["method"] == "profile"]
    for name in metrics:
        assert int((profile_df["metric_name"] == name).sum()) == 1


def test_profile_resume_complete_skips(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=[
            "resume=true",
            "eval.skip_linear=true",
            "eval.profile.enabled=true",
            "eval.profile.n_warmup=1",
            "eval.profile.n_measure=1",
        ],
    )

    seed_rows = [_resume_row(cfg, method="knn5", metric_name="accuracy")]
    for name in _profile_metric_names(cfg.eval.profile):
        seed_rows.append(_resume_row(cfg, method="profile", metric_name=name))
    pd.DataFrame(seed_rows).to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.measure_profile") as profile_mock,
    ):
        main.__wrapped__(cfg)

    profile_mock.assert_not_called()
