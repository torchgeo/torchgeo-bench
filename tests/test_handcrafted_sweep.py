"""Tests for the handcrafted classification sweep's coverage and output checks."""

from experiments.run_handcrafted import (
    build_jobs,
    classification_datasets,
    model_name,
    validate_case,
)


def test_classification_inventory_excludes_other_tasks():
    names = classification_datasets()
    assert len(names) == 13
    assert {"benv2", "treesatai", "m-bigearthnet", "eurosat-spatial", "resisc45"}.issubset(names)
    assert not {"burn_scars", "pastis", "spacenet2"}.intersection(names)


def test_four_levels_produce_52_independent_classification_jobs():
    names = classification_datasets()
    jobs = [job for level in range(4) for job in build_jobs(level, names)]
    assert len(jobs) == 52
    assert len({job.label for job in jobs}) == 52
    for job in jobs:
        assert "dataset.bands=all" in job.overrides
        assert "dataset.normalization=identity" in job.overrides
        assert not any("merge_val" in value or "image_size" in value for value in job.overrides)
    assert model_name(0) == "imagestats_handcrafted_control"
    assert "model.level=3" in build_jobs(3, ["eurosat"])[0].overrides
    assert "model=handcrafted_level3" in build_jobs(3, ["eurosat"])[0].overrides


def _row(method):
    return {
        "dataset": "toy",
        "method": method,
        "metric_name": "micro_mAP",
        "bands": "all",
        "normalization": "identity",
        "merge_val": "True",
        "feature_dim": "50",
        "num_classes": "19",
        "image_size": "224",
        "seed": "0",
        "config_hash": "current",
        "interpolation": "bilinear",
        "partition": "default",
        "bootstrap": "200",
        "c_range_start": "-6",
        "c_range_stop": "4",
        "c_range_num": "40",
        "n_train": "20",
        "n_val": "4",
        "n_test": "4",
        "metric_value": "0.6",
        "ci_lower": "0.5",
        "ci_upper": "0.7",
    }


def _spec():
    return {
        "multilabel": True,
        "feature_dim": 50,
        "num_classes": 19,
        "config_hashes": {"current": "cuda:0"},
        "split_sizes": {"train": 20, "val": 4, "test": 4},
    }


def test_success_requires_both_methods_not_just_a_successful_process():
    assert not validate_case([_row("knn5"), _row("linear")], "toy", _spec())
    assert "expected one row" in validate_case([_row("knn5")], "toy", _spec())[0]
    assert validate_case([_row("knn5"), _row("linear"), _row("linear")], "toy", _spec())


def test_other_resume_configurations_do_not_count_as_current_rows():
    older = {**_row("linear"), "config_hash": "different-device"}
    assert not validate_case([_row("knn5"), _row("linear"), older], "toy", _spec())
    assert validate_case([_row("knn5"), older], "toy", _spec())


def test_wrong_settings_scores_and_feature_width_are_visible():
    for field, value in (
        ("normalization", "bandspec_zscore"),
        ("feature_dim", "49"),
        ("metric_name", "accuracy"),
        ("merge_val", "False"),
        ("n_train", "19"),
        ("metric_value", "nan"),
        ("bootstrap", "0"),
        ("c_range_num", "1"),
        ("ci_lower", "nan"),
    ):
        rows = [_row("knn5"), {**_row("linear"), field: value}]
        assert validate_case(rows, "toy", _spec())
