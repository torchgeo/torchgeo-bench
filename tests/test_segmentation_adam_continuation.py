"""CPU regression coverage for optional Adam decay and guarded terminal continuation."""

import copy
import importlib
import json
import shutil
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
f = importlib.import_module("_segmentation_features")
c = importlib.import_module("_segmentation_convergence")
s = importlib.import_module("_segmentation_study")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def cpu_rng(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    rng = c.rng_snapshot(CPU)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    backend = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    c.restore_rng(rng, CPU)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    ) = backend


@pytest.fixture
def scratch() -> Iterator[Path]:
    path = ROOT / f".study-test-{uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def payload() -> dict:
    generator = torch.Generator().manual_seed(31)
    spec = {
        "dataset": "toy",
        "model": "tiny",
        "bands": "rgb",
        "normalization": "identity",
        "num_classes": 2,
        "image_size": 4,
        "split_counts": {"train": 5, "val": 3, "test": 2},
        "layers": ["deep", "middle", "early", "shallow"],
        "strict_determinism": True,
        "versions": {"python": "test", "packages": {}, "source_sha256": "old-source"},
    }
    splits = {
        split: {
            "features": [torch.randn(n, 2, 2, 2, generator=generator) for _ in spec["layers"]],
            "masks": torch.randint(2, (n, 4, 4), generator=generator),
        }
        for split, n in spec["split_counts"].items()
    }
    return {
        "splits": splits,
        "metadata": {
            "spec": spec,
            "key": f.identity(spec),
            "feature_shapes": [[2, 2, 2]] * 4,
            "tensor_sha256": f.tensor_digest(f.cache_tensors(splits)),
            "backbone_state_sha256": "same-frozen-backbone",
        },
    }


@pytest.fixture
def config() -> c.TrialConfig:
    return c.TrialConfig(
        "linear",
        "adam",
        0.03,
        batch_size=2,
        hidden_dim=4,
        check_every=1,
        patience=3,
        min_iterations=0,
        absolute_tol=1e-4,
        relative_tol=0,
        bootstrap=4,
    )


def selected(payload: dict) -> tuple[dict, dict]:
    splits, metadata = f.select_layers(payload, ["deep"])
    return {name: f.device_cache(data, CPU) for name, data in splits.items()}, metadata


def fresh_metadata(metadata: dict) -> dict:
    result = copy.deepcopy(metadata)
    result["spec"]["versions"]["source_sha256"] = "new-source"
    result["key"] = f.identity(result["spec"])
    return result


def create_parent(
    output: Path, payload: dict, config: c.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Create a schema-2 artifact with real Adam momentum and controlled stopping diagnostics."""
    caches, metadata = selected(payload)
    identity = c.scientific_identity(config, metadata)
    identity["schema"] = 2
    identity.pop("continuation")
    for name in c.SCHEDULE_FIELDS:
        identity["trial"].pop(name)
    directory = c.trial_directory(output / "trials", identity)
    directory.mkdir(parents=True)
    f.atomic_json(directory / "identity.json", identity)
    f.seed_everything(config.seed)
    run = c.ConvergenceRun(
        f.make_head("linear", caches["train"], 2, 4),
        caches,
        config,
        c.TrialStore(directory, identity),
    )
    losses = iter([0.5, 2.0, 1.0, 2.0])
    with monkeypatch.context() as context:
        context.setattr(run.objective, "monitor", lambda **_kwargs: (next(losses), None))
        result = run.fit()
    assert result["stop_reason"] == "no_best_improvement"
    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    state.pop("scheduler")
    state.pop("continuation")
    for row in [*state["curve"], state["best"]]:
        for name in ("adam_schedule", "learning_rate", "next_learning_rate", "lr_decay_count"):
            row.pop(name)
    result["final"] = state["curve"][-1]
    result["selected"] = {
        k: v for k, v in state["best"].items() if k not in ("state_dict", "buffers", "modes")
    }
    result.update(identity=identity, trial_key=directory.name, cache_metadata=metadata)
    f.atomic_save(directory / "checkpoint.pt", state)
    f.atomic_json(directory / "result.json", {**result, "result_sha256": f.identity(result)})
    return {
        "output": output,
        "directory": directory,
        "result": result,
        "state": state,
        "caches": caches,
        "metadata": metadata,
        "config": config,
    }


@pytest.fixture
def parent(
    scratch: Path, payload: dict, config: c.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> dict:
    return create_parent(scratch / "parent", payload, config, monkeypatch)


def continuation(parent: dict) -> tuple[c.TrialConfig, dict]:
    config = replace(parent["config"], adam_schedule="plateau")
    provenance, _ = c.inspect_continuation_parent(parent["directory"], config, ["deep"])
    metadata = fresh_metadata(parent["metadata"])
    metadata["continuation"] = provenance
    return config, metadata


def new_run(parent: dict, config: c.TrialConfig) -> c.ConvergenceRun:
    f.seed_everything(config.seed)
    return c.ConvergenceRun(
        f.make_head(config.head, parent["caches"]["train"], 2, config.hidden_dim),
        parent["caches"],
        config,
    )


def assert_tree(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_tree(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        for a, b in zip(actual, expected, strict=True):
            assert_tree(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("schedule", ["constant", "plateau"])
def test_decay_only_changes_opt_in_and_restarts_full_window(
    schedule: str, payload: dict, config: c.TrialConfig
) -> None:
    caches, _ = selected(payload)
    config = replace(config, adam_schedule=schedule, min_iterations=50)
    run = c.ConvergenceRun(f.make_head("linear", caches["train"], 2, 4), caches, config)
    for iteration, loss in enumerate([0.5, 2, 1, 2], start=50):
        run.counters["epochs"] = iteration
        reason = run.stopping_reason(loss, None, changed=True)
    if schedule == "constant":
        assert reason == "no_best_improvement"
        assert run.current_lr == config.lr
        assert run.scheduler.decay_count == 0
    else:
        assert reason is None
        assert run.current_lr == pytest.approx(config.lr * config.adam_lr_factor)
        assert run.scheduler.decay_count == 1
        assert run.plateau.recent_losses == []
        assert run.plateau.reference == 2
        for iteration in (54, 55):
            run.counters["epochs"] = iteration
            assert run.stopping_reason(2, None, changed=True) is None
        run.counters["epochs"] = 56
        assert run.stopping_reason(2, None, changed=True) == "train_loss_plateau"


def test_real_cpu_annealed_fit_reaches_flat_loss_after_two_decays() -> None:
    f.seed_everything(12)
    images = torch.randn(6, 2, 2, 2)
    masks = (images[:, 0] > 0).long().repeat_interleave(2, 1).repeat_interleave(2, 2)
    cache = f.GPUTensorCache([images], masks, CPU)
    config = c.TrialConfig(
        "linear",
        "adam",
        0.3,
        batch_size=2,
        hidden_dim=4,
        check_every=1,
        min_iterations=0,
        patience=3,
        absolute_tol=1e-4,
        relative_tol=0,
        max_iterations=60,
        bootstrap=2,
        adam_schedule="plateau",
    )
    run = c.ConvergenceRun(
        f.make_head("linear", cache, 2, 4), dict.fromkeys(f.SPLITS, cache), config
    )
    result = run.fit()
    assert result["status"] == "converged"
    assert result["stop_reason"] == "train_loss_plateau"
    assert result["lr_decay_count"] == 2
    assert result["iterations"] < 60
    assert result["current_lr"] == pytest.approx(0.003)
    assert result["final"]["recent_ce_range"] <= config.absolute_tol
    assert result["stationary"] is False


def test_decay_resume_preserves_optimizer_rng_and_window(
    payload: dict, config: c.TrialConfig, scratch: Path
) -> None:
    caches, metadata = selected(payload)
    config = replace(config, adam_schedule="plateau")
    identity = c.scientific_identity(config, metadata)
    store = c.TrialStore(scratch, identity)
    run = c.ConvergenceRun(f.make_head("linear", caches["train"], 2, 4), caches, config, store)
    run.block(initial=True)
    run.block()
    assert run.decay_learning_rate(run.curve[-1]["train_ce"])
    run.block()
    saved = copy.deepcopy(store.load())
    assert saved["scheduler"]["decay_count"] == 1
    assert saved["plateau"]["recent_losses"]
    run.store = None
    run.block()
    expected = copy.deepcopy(run.state())
    resumed = c.ConvergenceRun(f.make_head("linear", caches["train"], 2, 4), caches, config)
    resumed.restore(saved)
    resumed.block()
    for key in ("model", "optimizer", "rng", "scheduler", "counters", "plateau"):
        assert_tree(resumed.state()[key], expected[key])


def test_floor_cap_and_nonfinite_never_claim_convergence(
    payload: dict, config: c.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    caches, _ = selected(payload)
    config = replace(config, adam_schedule="plateau", adam_min_lr=config.lr)
    run = c.ConvergenceRun(f.make_head("linear", caches["train"], 2, 4), caches, config)
    for iteration, loss in enumerate([0.5, 2, 1, 2]):
        run.counters["epochs"] = iteration
        reason = run.stopping_reason(loss, None, changed=True)
    assert reason == "lr_floor"
    assert run.scheduler.decay_count == 0
    capped = c.ConvergenceRun(
        f.make_head("linear", caches["train"], 2, 4),
        caches,
        replace(config, adam_min_lr=1e-8, max_iterations=1),
    )
    result = capped.fit()
    assert result["status"] == "not_converged"
    assert result["stop_reason"] == "safety_max_iterations"
    monkeypatch.setattr(run.objective, "monitor", Mock(side_effect=f.DivergedError("nonfinite")))
    run.block(initial=True)
    assert run.status == "failed"
    assert run.stop_reason == "nonfinite"
    assert run.scheduler.decay_count == 0


def test_schedule_does_not_decay_during_material_recovery(
    payload: dict, config: c.TrialConfig
) -> None:
    caches, _ = selected(payload)
    config = replace(config, adam_schedule="plateau")
    run = c.ConvergenceRun(f.make_head("linear", caches["train"], 2, 4), caches, config)
    for iteration, loss in enumerate([0.5, 35.7, 5.8, 2.91, 1.2, 0.9]):
        run.counters["epochs"] = iteration
        assert run.stopping_reason(loss, None, changed=True) is None
    assert run.scheduler.decay_count == 0


def test_warm_start_retains_parent_optimizer_rng_best_and_cumulative_time(parent: dict) -> None:
    config, metadata = continuation(parent)
    before = {p.name: f.file_digest(p) for p in parent["directory"].iterdir() if p.is_file()}
    run = new_run(parent, config)
    c.warm_continue(run, metadata)
    state = parent["state"]
    assert_tree(c.model_snapshot(run.head), state["model"])
    assert_tree(run.optimizer.state_dict()["state"], state["optimizer"]["state"])
    assert_tree(c.rng_snapshot(CPU), state["rng"])
    assert_tree(run.best["state_dict"], state["best"]["state_dict"])
    assert run.best["val_miou"] == parent["result"]["selected"]["val_miou"]
    assert run.counters == state["counters"]
    assert run.current_lr == pytest.approx(config.lr * 0.1)
    assert run.status == "running"
    assert run.stop_reason is None
    assert run.plateau.recent_losses == []
    assert run.summary()["incremental_optimization_seconds"] == 0
    assert run.summary()["selected_from_parent"] is True
    assert (
        run.timings.seconds["optimization"] == parent["result"]["timings_seconds"]["optimization"]
    )
    assert before == {
        p.name: f.file_digest(p) for p in parent["directory"].iterdir() if p.is_file()
    }


def test_continuation_uses_new_checkpoint_for_resume_and_keeps_original_best(
    parent: dict, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, metadata = continuation(parent)
    output = scratch / "child" / "trials"
    identity = c.scientific_identity(config, metadata)
    directory = c.trial_directory(output, identity)
    directory.mkdir(parents=True)
    f.atomic_json(directory / "identity.json", identity)
    run = new_run(parent, config)
    run.store = c.TrialStore(directory, identity)
    c.warm_continue(run, metadata)
    monkeypatch.setattr(c.FullObjective, "monitor", lambda *_args, **_kwargs: (2.0, None))
    evaluate = f.evaluate

    def worse_validation(head: torch.nn.Module, cache: f.GPUTensorCache, batch: int) -> dict:
        value = evaluate(head, cache, batch)
        if cache is parent["caches"]["val"]:
            value["miou"] = 0.0
        return value

    monkeypatch.setattr(f, "evaluate", worse_validation)
    run.block()
    assert run.plateau.stale_checks == 1
    saved = run.store.load()
    parent_hashes = {p.name: f.file_digest(p) for p in parent["directory"].iterdir() if p.is_file()}
    monkeypatch.setattr(
        c, "warm_continue", Mock(side_effect=AssertionError("use the child's checkpoint"))
    )
    result = c.run_trial(config, parent["caches"], metadata, output, resume=True)
    assert result["status"] == "converged"
    assert result["stop_reason"] == "train_loss_plateau"
    assert result["lr_decay_count"] == 1
    assert result["counters"]["epochs"] == parent["result"]["counters"]["epochs"] + 3
    assert result["selected_from_parent"] is True
    assert result["selected"]["val_miou"] == parent["result"]["selected"]["val_miou"]
    assert result["incremental_optimization_seconds"] > 0
    assert result["timings_seconds"]["optimization"] == pytest.approx(
        parent["result"]["timings_seconds"]["optimization"]
        + result["incremental_optimization_seconds"]
    )
    assert result["incremental_training_wall_seconds"] > 0
    checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["identity"] != parent["state"]["identity"]
    assert checkpoint["continuation"] == saved["continuation"]
    assert_tree(checkpoint["best"]["state_dict"], parent["state"]["best"]["state_dict"])
    assert parent_hashes == {
        p.name: f.file_digest(p) for p in parent["directory"].iterdir() if p.is_file()
    }
    assert "next_learning_rate" in (directory / "curve.csv").read_text()
    job = s.Job(config, "deep", ("deep",), metadata["continuation"])
    row = s.result_row(job, metadata, result)
    assert row["adam_schedule"] == "plateau"
    assert row["lr"] == 0.03
    assert row["current_lr"] == pytest.approx(0.003)
    assert row["selected_learning_rate"] == 0.03
    assert row["parent_checkpoint_sha256"] == metadata["continuation"]["checkpoint_file_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", 1), ("lr", 0.02), ("batch_size", 3), ("absolute_tol", 1e-5), ("bootstrap", 5)],
)
def test_continuation_rejects_changed_fit_settings(parent: dict, field: str, value: float) -> None:
    config, _ = continuation(parent)
    with pytest.raises(ValueError, match="scientific trial settings mismatch"):
        c.inspect_continuation_parent(
            parent["directory"], replace(config, **{field: value}), ["deep"]
        )


@pytest.mark.parametrize(
    "field",
    [
        "dataset",
        "model",
        "normalization",
        "num_classes",
        "split_counts",
        "tensor_sha256",
        "feature_shapes",
        "selected_feature_shapes",
        "selected_layers",
        "backbone_state_sha256",
        "dependencies",
    ],
)
def test_continuation_rejects_changed_data_backbone_or_features(parent: dict, field: str) -> None:
    config, metadata = continuation(parent)
    if field == "dependencies":
        metadata["spec"]["versions"]["packages"]["torch"] = "different"
    elif field in metadata["spec"]:
        metadata["spec"][field] = "different"
    else:
        metadata[field] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        c.verified_continuation_parent(metadata["continuation"], config, metadata)


def test_source_evolution_is_allowed_only_with_explicit_continuation(
    parent: dict, payload: dict, scratch: Path
) -> None:
    config, metadata = continuation(parent)
    assert (
        c.verified_continuation_parent(metadata["continuation"], config, metadata)["identity"][
            "schema"
        ]
        == 2
    )
    path = scratch / "old-cache.pt"
    f.atomic_save(path, payload)
    with pytest.raises(ValueError, match="Cache configuration mismatch"):
        f.load_cache(path, metadata["spec"])
    identity = c.scientific_identity(config, metadata)
    with pytest.raises(ValueError, match="Checkpoint identity mismatch"):
        c.TrialStore(parent["directory"], identity).load()


@pytest.mark.parametrize("damage", ["identity", "initial", "optimizer", "order", "nonfinite"])
def test_warm_start_rejects_unpaired_or_nonfinite_checkpoint(parent: dict, damage: str) -> None:
    state = copy.deepcopy(parent["state"])
    if damage == "identity":
        state["identity"]["schema"] = 1
    elif damage == "initial":
        state["initial_state_sha256"] = "wrong"
    elif damage == "optimizer":
        state["optimizer"]["param_groups"][0]["weight_decay"] = 0.2
    elif damage == "order":
        state["model"]["state_dict"] = dict(reversed(state["model"]["state_dict"].items()))
    else:
        next(iter(state["model"]["state_dict"].values())).fill_(float("nan"))
    f.atomic_save(parent["directory"] / "checkpoint.pt", state)
    config, metadata = continuation(parent)
    with pytest.raises(
        (ValueError, f.DivergedError), match=r"mismatch|matching terminal|Nonfinite"
    ):
        c.warm_continue(new_run(parent, config), metadata)


def test_warm_start_rejects_changed_initialization_and_postmapping_checksum(parent: dict) -> None:
    config, metadata = continuation(parent)
    run = new_run(parent, config)
    run.initial_state_sha256 = "unpaired-initialization"
    with pytest.raises(ValueError, match="initial decoder state mismatch"):
        c.warm_continue(run, metadata)
    with (parent["directory"] / "checkpoint.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="provenance checksum mismatch"):
        c.warm_continue(new_run(parent, config), metadata)


@pytest.mark.parametrize("reason", ["nonfinite", "safety_max_iterations", "lr_floor"])
def test_continuation_refuses_other_terminal_reasons(parent: dict, reason: str) -> None:
    result = copy.deepcopy(parent["result"])
    result["stop_reason"] = reason
    f.atomic_json(
        parent["directory"] / "result.json", {**result, "result_sha256": f.identity(result)}
    )
    with pytest.raises(ValueError, match="must be uncapped"):
        continuation(parent)


def test_continuation_refuses_a_capped_parent_even_when_cap_did_not_stop_it(
    scratch: Path, payload: dict, config: c.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = create_parent(
        scratch / "capped", payload, replace(config, max_iterations=10), monkeypatch
    )
    with pytest.raises(ValueError, match="must be uncapped"):
        continuation(parent)


def test_cli_pairing_is_unique_readonly_and_uses_parent_union(parent: dict, scratch: Path) -> None:
    config, _ = continuation(parent)
    job = s.Job(config, "deep", ("deep",))
    jobs, spec = s.continuation_jobs([job], parent["output"], scratch / "child")
    assert spec["layers"] == ["deep", "middle", "early", "shallow"]
    assert jobs[0].continuation["parent_identity"] == parent["result"]["identity"]
    with pytest.raises(ValueError, match="new output directory"):
        s.continuation_jobs([job], parent["output"], parent["output"])
    with pytest.raises(ValueError, match="requires --phase adam"):
        s.continuation_jobs(
            [replace(job, trial=replace(config, adam_schedule="constant"))],
            parent["output"],
            scratch / "other",
        )
    with pytest.raises(ValueError, match="found 0"):
        s.continuation_jobs(
            [replace(job, trial=replace(config, seed=12))], parent["output"], scratch / "other"
        )
    shutil.copytree(parent["directory"], parent["output"] / "trials" / "duplicate")
    with pytest.raises(ValueError, match="found 2"):
        s.continuation_jobs([job], parent["output"], scratch / "other")


def test_cli_default_constant_and_optional_schedule_stays_adam_only() -> None:
    parser = s.build_parser("optimizer", "")
    assert parser.parse_args([]).adam_schedule == "constant"
    args = parser.parse_args(["--adam-schedule", "plateau"])
    jobs = s.build_jobs(args, "optimizer")
    assert all(
        job.trial.adam_schedule == "plateau" for job in jobs if job.trial.optimizer == "adam"
    )
    assert all(
        job.trial.adam_schedule == "constant" for job in jobs if job.trial.optimizer == "lbfgs"
    )
    with pytest.raises(ValueError, match="only for Adam"):
        c.TrialConfig("linear", "lbfgs", 1, adam_schedule="plateau")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adam_lr_factor", 1),
        ("adam_lr_factor", 0),
        ("adam_lr_factor", float("nan")),
        ("adam_min_lr", 0),
        ("adam_min_lr", float("inf")),
        ("adam_schedule", "unknown"),
    ],
)
def test_invalid_schedule_controls_are_rejected(
    config: c.TrialConfig, field: str, value: object
) -> None:
    with pytest.raises(ValueError, match=r"adam_|schedule"):
        replace(config, **{field: value})


def test_lr_floor_is_a_terminal_nonconverged_result(
    payload: dict, config: c.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    caches, _ = selected(payload)
    config = replace(config, adam_schedule="plateau", adam_min_lr=config.lr)
    run = c.ConvergenceRun(f.make_head("linear", caches["train"], 2, 4), caches, config)
    losses = iter([0.5, 2, 1, 2])
    monkeypatch.setattr(run.objective, "monitor", lambda **_kwargs: (next(losses), None))
    result = run.fit()
    assert result["status"] == "not_converged"
    assert result["stop_reason"] == "lr_floor"
    assert result["stationary"] is False
    assert result["lr_decay_count"] == 0


def test_warm_continuation_marks_mixed_hardware(
    parent: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, metadata = continuation(parent)
    monkeypatch.setattr(
        f, "hardware", lambda _device: {"device": "cuda:4", "name": "different-GPU"}
    )
    run = new_run(parent, config)
    c.warm_continue(run, metadata)
    assert run.hardware_history[0] == parent["state"]["hardware_history"][0]
    assert run.hardware_history[-1]["device"] == "cuda:4"
    assert run.summary()["mixed_hardware_timings"] is True


def test_cpu_worker_manifest_continues_and_resumes_legacy_parent(
    payload: dict, config: c.TrialConfig, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload["metadata"]["spec"]["versions"] = f.software_versions()
    payload["metadata"]["spec"]["versions"]["source_sha256"] = "legacy-source"
    payload["metadata"]["key"] = f.identity(payload["metadata"]["spec"])
    parent = create_parent(scratch / "parent", payload, config, monkeypatch)
    config, metadata = continuation(parent)
    payload["metadata"]["spec"]["versions"] = f.software_versions()
    payload["metadata"]["key"] = f.identity(payload["metadata"]["spec"])
    cache = scratch / "fresh-features.pt"
    f.atomic_save(cache, payload)
    job = s.Job(config, "deep", ("deep",), metadata["continuation"])
    manifest = {
        "job": s.asdict(job),
        "cache_spec": payload["metadata"]["spec"],
        "cache_tensor_sha256": payload["metadata"]["tensor_sha256"],
    }
    manifest_path = scratch / "job.json"
    f.atomic_json(manifest_path, manifest)
    output = scratch / "child"
    args = s.build_parser("optimizer", "").parse_args(
        [
            "--worker",
            str(manifest_path),
            "--cache",
            str(cache),
            "--device",
            "cpu",
            "--output-dir",
            str(output),
        ]
    )
    monkeypatch.setattr(c.FullObjective, "monitor", lambda *_args, **_kwargs: (2.0, None))
    s.run_worker(args)
    directory = c.trial_directory(output / "trials", s.job_identity(job, payload["metadata"]))
    before = (directory / "result.json").read_bytes()
    result = json.loads(before)
    assert result["status"] == "converged"
    assert result["adam_schedule"] == "plateau"
    assert result["current_lr"] == pytest.approx(0.003)
    args.resume = True
    s.run_worker(args)
    assert (directory / "result.json").read_bytes() == before
    manifest["cache_spec"]["versions"]["source_sha256"] = "unexpected-worker-source"
    f.atomic_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="source/dependency identity changed"):
        s.run_worker(args)


def test_shared_existing_cache_is_readonly_even_when_source_mismatches(
    payload: dict, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scratch / "features.pt"
    f.atomic_save(path, payload)
    before = f.file_digest(path)
    monkeypatch.setattr(
        f, "FileLock", Mock(side_effect=AssertionError("immutable readers need no producer lock"))
    )
    loaded = f.prepare_cache(path, payload["metadata"]["spec"], CPU, 0)
    assert loaded["metadata"]["tensor_sha256"] == payload["metadata"]["tensor_sha256"]
    changed = fresh_metadata(payload["metadata"])["spec"]
    with pytest.raises(ValueError, match="Cache configuration mismatch"):
        f.prepare_cache(path, changed, CPU, 0)
    assert f.file_digest(path) == before
    assert not Path(str(path) + ".lock").exists()
