"""CPU-only coverage of the standalone optimizer and layer production studies."""

import copy
import csv
import importlib
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
features = importlib.import_module("_segmentation_features")
convergence = importlib.import_module("_segmentation_convergence")
study = importlib.import_module("_segmentation_study")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def cpu_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    state = convergence.rng_snapshot(CPU)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    )
    matmul = torch.backends.cuda.matmul.allow_tf32
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    features.seed_everything(13)
    yield
    convergence.restore_rng(state, CPU)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.allow_tf32,
    ) = cudnn
    torch.backends.cuda.matmul.allow_tf32 = matmul


@pytest.fixture
def scratch() -> Iterator[Path]:
    directory = ROOT / "outputs" / f"production-study-tests-{uuid4().hex}"
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)


@pytest.fixture
def payload() -> dict:
    generator = torch.Generator().manual_seed(12)
    spec = {
        "dataset": "synthetic",
        "model": "tiny",
        "bands": "rgb",
        "normalization": "identity",
        "split_counts": {"train": 5, "val": 3, "test": 2},
        "image_size": 8,
        "num_classes": 2,
        "layers": ["deep", "middle", "earlier", "shallow"],
        "strict_determinism": True,
        "versions": {},
    }
    splits = {
        split: {
            "features": [torch.randn(n, 4, 2, 2, generator=generator) for _ in range(4)],
            "masks": torch.randint(2, (n, 8, 8), generator=generator),
        }
        for split, n in spec["split_counts"].items()
    }
    splits["train"]["masks"][2:4] = 255
    return {
        "metadata": {
            "spec": spec,
            "key": features.identity(spec),
            "tensor_sha256": features.tensor_digest(features.cache_tensors(splits)),
            "feature_shapes": [[4, 2, 2]] * 4,
        },
        "splits": splits,
    }


@pytest.fixture
def config() -> convergence.TrialConfig:
    return convergence.TrialConfig(
        "linear",
        "adam",
        0.01,
        batch_size=2,
        hidden_dim=4,
        check_every=1,
        lbfgs_iterations=2,
        max_iterations=2,
        bootstrap=4,
    )


def selected(payload: dict, layers: list[str] | None = None) -> tuple[dict, dict]:
    data, metadata = features.select_layers(
        payload, layers or payload["metadata"]["spec"]["layers"]
    )
    return {name: features.device_cache(value, CPU) for name, value in data.items()}, metadata


def runner_for(scratch: Path, payload: dict, jobs: list[study.Job]) -> study.StudyRunner:
    config = study.StudyConfig(
        root=ROOT,
        cli=Path(sys.executable),
        state_dir=scratch / "state",
        gpus=[0, 1],
        num_workers=0,
        max_attempts=1,
        script=ROOT / "scripts/run_segmentation_optimizer_study.py",
        cache=scratch / "features.pt",
        output_dir=scratch,
        resume=True,
    )
    runner = study.StudyRunner(config, jobs, payload["metadata"]["spec"], payload["metadata"])
    runner.log_dir.mkdir(parents=True)
    return runner


def assert_trees_equal(actual: object, expected: object) -> None:
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_trees_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            assert_trees_equal(a, b)
    else:
        assert actual == expected


def test_safe_cache_roundtrip_and_union_subset_invariants(payload: dict, scratch: Path) -> None:
    path = scratch / "cache.pt"
    features.atomic_save(path, payload)
    loaded = features.load_cache(path, payload["metadata"]["spec"])
    data, metadata = features.select_layers(loaded, ["shallow", "middle"])
    assert data["train"]["features"][0] is loaded["splits"]["train"]["features"][3]
    assert data["test"]["masks"] is loaded["splits"]["test"]["masks"]
    assert metadata["selected_layers"] == ["shallow", "middle"]
    assert metadata["key"] == loaded["metadata"]["key"]
    assert metadata["tensor_sha256"] == loaded["metadata"]["tensor_sha256"]
    with pytest.raises(ValueError, match="absent"):
        features.select_layers(loaded, ["unknown"])
    with pytest.raises(ValueError, match="Duplicate"):
        features.select_layers(loaded, ["deep", "deep"])
    loaded["splits"]["train"]["features"][0][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum"):
        features.validate_cache(loaded, payload["metadata"]["spec"])
    with pytest.raises(ValueError, match="configuration mismatch"):
        features.load_cache(path, {**payload["metadata"]["spec"], "image_size": 12})


@pytest.mark.parametrize("corruption", ["splits", "counts", "labels", "geometry", "dtype"])
def test_cache_rejects_invalid_invariants(payload: dict, corruption: str) -> None:
    if corruption == "splits":
        del payload["splits"]["test"]
    elif corruption == "counts":
        payload["splits"]["train"]["masks"] = payload["splits"]["train"]["masks"][:1]
    elif corruption == "labels":
        payload["splits"]["train"]["masks"].fill_(8)
    elif corruption == "geometry":
        payload["splits"]["val"]["features"][0] = torch.ones(3, 4, 3, 3)
    else:
        payload["splits"]["val"]["features"][0] = payload["splits"]["val"]["features"][0].half()
    with pytest.raises(ValueError, match=r"Cache must|Invalid|geometry"):
        features.validate_cache(payload, payload["metadata"]["spec"])


def test_extractor_uses_actual_inputs_once_and_no_dummy_forward() -> None:
    class Backbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = nn.Sequential(nn.Identity(), nn.Identity())
            self.calls = 0

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            return self.backbone(image)

    backbone = Backbone()
    extractor = features.FeatureExtractor(backbone, ["1", "0"])
    assert backbone.calls == 0
    samples = [
        {"image": torch.ones(2, 3, 12, 12) * i, "mask": torch.zeros(1, 12, 12)} for i in range(3)
    ]
    result = extractor.extract(DataLoader(samples, batch_size=2))
    assert backbone.calls == 2
    assert result["features"][0].shape == (3, 3, 12, 12)
    assert result["masks"].shape == (3, 12, 12)
    assert_trees_equal(result["features"][0], result["features"][1])
    for hook in extractor.hooks:
        hook.remove()


def test_grid_defaults_union_and_explicit_compatibility() -> None:
    args = study.build_parser("optimizer", "").parse_args([])
    jobs = study.build_jobs(args, "optimizer")
    assert len(jobs) == 120
    assert all(job.trial.optimizer == "adam" for job in jobs[:90])
    assert all(job.trial.optimizer == "lbfgs" for job in jobs[90:])
    assert args.max_iterations is None
    assert args.lbfgs_iterations == 20
    assert args.history_size == 10
    assert all(len(job.layers) == 1 for job in jobs if job.trial.head == "patch_linear")
    args = study.build_parser("layer", "").parse_args([])
    jobs = study.build_jobs(args, "layer")
    assert len(jobs) == 189
    assert set(study.layer_union(jobs)) == {f"blocks.{n}" for n in (11, 10, 9, 8, 5, 4, 3, 2)}
    assert len(study.layer_union(jobs)) == 8
    for head in ("dpt", "patch_linear"):
        args.heads = [head]
        with pytest.raises(ValueError, match="requires"):
            study.build_jobs(args, "layer")
    args = study.build_parser("layer", "").parse_args(
        ["--heads", "dpt", "--layer-groups", "late_four", "spread_four", "early_four"]
    )
    assert len(study.build_jobs(args, "layer")) == 27
    args = study.build_parser("layer", "").parse_args(
        ["--heads", "patch_linear", "--layer-groups", "deep_single"]
    )
    assert len(study.build_jobs(args, "layer")) == 9


@pytest.mark.parametrize(
    "option",
    [
        ["--seeds", "0", "0"],
        ["--heads", "linear", "linear"],
        ["--adam-lrs", "0.001", "0.001"],
        ["--adam-lrs", "nan"],
        ["--layer-groups-json", '{"a":["x"],"b":["x"]}'],
        ["--layer-groups-json", '{"a":["x","x"]}'],
        ["--layer-groups-json", '{"a":["x"],"a":["y"]}'],
    ],
)
def test_grid_rejects_ambiguous_or_invalid_inputs(option: list[str]) -> None:
    args = study.build_parser("layer", "").parse_args(option)
    with pytest.raises(ValueError, match=r"[Dd]uplicate|positive"):
        study.build_jobs(args, "layer")


def test_gpu_autodetection_duplicates_and_command(
    scratch: Path, payload: dict, config: convergence.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 6)
    assert study.selected_gpus("all") == list(range(6))
    assert study.selected_gpus("1,4") == [1, 4]
    with pytest.raises(ValueError, match="Duplicate"):
        study.selected_gpus("2,2")
    with pytest.raises(ValueError, match="invalid"):
        study.selected_gpus("6")
    job = study.Job(config, "deep", ("deep",))
    runner = runner_for(scratch, payload, [job])
    command = runner._command(job, 4, 1)
    assert command[0] == sys.executable
    assert Path(command[1]).parts[-2:] == ("scripts", "run_segmentation_optimizer_study.py")
    assert command[command.index("--device") + 1] == "cuda:4"
    assert "--worker" in command
    assert "--resume" in command
    assert command[command.index("--worker") + 1].endswith(f"{job.job_id}.json")


@pytest.mark.parametrize("suite", ["optimizer", "layer"])
def test_dry_run_does_not_write_or_launch(
    suite: str, scratch: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(features, "prepare_cache", Mock(side_effect=AssertionError("no cache")))
    monkeypatch.setattr(
        study.StudyRunner, "_run_attempt", Mock(side_effect=AssertionError("no worker"))
    )
    output = scratch / "untouched"
    caplog.set_level(logging.INFO)
    args = [
        "--dry-run",
        "--heads",
        "linear",
        "--seeds",
        "0",
        "--adam-lrs",
        "0.001",
        "--output-dir",
        str(output),
    ]
    study.main(suite, "", args)
    assert not output.exists()
    assert "--worker" in caplog.text
    assert "cuda:0" in caplog.text


@pytest.mark.parametrize("head_name", features.HEADS)
@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_all_heads_optimize_pair_initializations_and_record_capacity(
    head_name: str, optimizer: str, payload: dict, config: convergence.TrialConfig
) -> None:
    if head_name == "dpt":
        pytest.importorskip("transformers", reason="The existing DPT head requires transformers")
    layers = ["deep"] if head_name == "patch_linear" else payload["metadata"]["spec"]["layers"]
    caches, metadata = selected(payload, layers)
    config = replace(config, head=head_name, optimizer=optimizer, lr=0.1)
    initial = []
    counts = []
    for lr in (0.01, 0.1):
        features.seed_everything(config.seed)
        head = features.make_head(head_name, caches["train"], 2, 4)
        run = convergence.ConvergenceRun(head, caches, replace(config, lr=lr))
        initial.append(run.initial_state_sha256)
        counts.append(sum(p.numel() for p in head.parameters()))
        assert run.optimize()
        result = run.summary()
        assert result["parameter_count"] == counts[-1]
        assert result["optimizer_settings"]["lr"] == lr
        if optimizer == "adam":
            assert type(run.optimizer) is torch.optim.Adam
            assert run.optimizer.defaults["weight_decay"] == 0
        else:
            assert run.optimizer.defaults["line_search_fn"] == "strong_wolfe"
            assert run.iterations <= 2
            assert run.objective.evaluations == run.lbfgs_state["func_evals"]
        if head_name == "patch_linear":
            assert head.patch_size == 4
    assert initial[0] == initial[1]
    assert counts[0] == counts[1]
    assert "hardware" not in convergence.scientific_identity(config, metadata)


def test_objective_is_deterministic_preserves_buffers(payload: dict) -> None:
    caches, _ = selected(payload)
    head = features.make_head("linear", caches["train"], 2, 4).eval()
    head.register_buffer("counter", torch.tensor(3), persistent=False)

    def count(module: nn.Module, _args: tuple, _output: torch.Tensor) -> None:
        module.counter.add_(1)

    head.register_forward_hook(count)
    objective = convergence.FullObjective(head, caches["train"], 2)
    before = convergence.model_snapshot(head)
    loss = objective()
    gradients = {name: p.grad.clone() for name, p in head.named_parameters()}
    assert_trees_equal(convergence.model_snapshot(head), before)
    assert_trees_equal(objective(), loss)
    assert_trees_equal({name: p.grad for name, p in head.named_parameters()}, gradients)


@pytest.mark.parametrize("head_name", ["linear", "conv_block", "fpn"])
@pytest.mark.parametrize("ignored_bucket", [False, True])
def test_fixed_bn_batch_gradient_mean_matches_full_objective_and_membership_persists(
    head_name: str, payload: dict, *, ignored_bucket: bool
) -> None:
    caches, _ = selected(payload)
    cache = caches["train"]
    for tensor in cache.layer_tensors:
        tensor.add_(torch.arange(5)[:, None, None, None] * 7)
        tensor[:, 0, 0, 0] = torch.arange(5)
    cache.masks[0, :3] = 255
    cache.masks[4, :5] = 255
    if not ignored_bucket:
        cache.masks[2:4] = torch.arange(128).reshape(2, 8, 8) % 2
    head = features.make_head(head_name, cache, 2, 4)
    assert any(isinstance(module, nn.BatchNorm2d) for module in head.modules())
    objective = convergence.FullObjective(head, cache, 2)
    objective()
    reference = {name: p.grad.clone() for name, p in head.named_parameters()}
    assert objective.batch_count == 3
    assert any(gradient.abs().max() > 0 for gradient in reference.values())

    # Real Adam with lr=0 keeps theta fixed while exposing real BN/CE gradients.
    optimizer = torch.optim.Adam(head.parameters(), lr=0.0)
    averaged = {name: torch.zeros_like(gradient) for name, gradient in reference.items()}
    buckets = []

    def capture_gradient(_optimizer: torch.optim.Optimizer, _args: tuple, _kwargs: dict) -> None:
        for name, p in head.named_parameters():
            averaged[name].add_(p.grad / objective.batch_count)

    def capture_membership(_module: nn.Module, args: tuple) -> None:
        buckets.append(tuple(args[0][0][:, 0, 0, 0].tolist()))

    gradient_hook = optimizer.register_step_pre_hook(capture_gradient)
    membership_hook = head.register_forward_pre_hook(capture_membership)
    counters = dict.fromkeys(("samples", "epochs", "optimizer_step_calls", "optimizer_updates"), 0)
    expected_buckets = {(0.0, 1.0), (4.0,)}
    if not ignored_bucket:
        expected_buckets.add((2.0, 3.0))
    orders = []
    for epoch in range(5):
        torch.manual_seed(epoch)
        buckets.clear()
        for value in averaged.values():
            value.zero_()
        convergence.adam_epoch(objective, optimizer, counters)
        assert len(buckets) == len(expected_buckets)
        assert set(buckets) == expected_buckets
        orders.append(tuple(buckets))
        for name, gradient in reference.items():
            torch.testing.assert_close(averaged[name], gradient, atol=1e-6, rtol=1e-4)
    gradient_hook.remove()
    membership_hook.remove()
    assert len(set(orders)) > 1
    assert counters["samples"] == 25
    assert counters["optimizer_updates"] == 5 * len(expected_buckets)
    assert all(state["step"] == counters["optimizer_updates"] for state in optimizer.state.values())


def test_stopping_plateau_accumulates_but_safety_and_stalls_are_not_convergence(
    payload: dict, config: convergence.TrialConfig
) -> None:
    plateau = convergence.Plateau()
    threshold = replace(config, min_iterations=0, patience=3, absolute_tol=0.01, relative_tol=0)
    assert not plateau.observe(1, 0, threshold)
    assert not plateau.observe(0.996, 1, threshold)
    assert not plateau.observe(0.992, 2, threshold)
    assert not plateau.observe(0.988, 3, threshold)
    assert plateau.stale_checks == 0
    assert not plateau.observe(0.984, 4, threshold)
    assert not plateau.observe(0.980, 5, threshold)
    assert not plateau.observe(0.976, 6, threshold)
    assert plateau.stale_checks == 0
    assert not plateau.observe(0.976, 7, threshold)
    assert not plateau.observe(0.976, 8, threshold)
    assert plateau.observe(0.976, 9, threshold) == "train_loss_plateau"
    caches, _ = selected(payload)
    head = features.make_head("linear", caches["train"], 2, 4)
    run = convergence.ConvergenceRun(head, caches, config)
    result = run.fit()
    assert result["status"] == "not_converged"
    assert result["stop_reason"] == "safety_max_iterations"
    assert result["stationary"] is False
    stalled = convergence.ConvergenceRun(head, caches, replace(config, optimizer="lbfgs"))
    stalled.counters["stalled_iteration_blocks"] = config.numerical_patience
    assert stalled.stopping_reason(1, 1, changed=False) == "numerical_stall"
    assert stalled.stopping_reason(1, 0, changed=False) == "gradient_tolerance"


@pytest.mark.parametrize("optimizer", ["adam", "lbfgs"])
def test_checkpoint_resume_replays_identically_and_repairs_partial_csv(
    optimizer: str, payload: dict, config: convergence.TrialConfig, scratch: Path
) -> None:
    config = replace(config, optimizer=optimizer, max_iterations=4, min_iterations=0)
    caches, metadata = selected(payload)
    expected = convergence.scientific_identity(config, metadata)
    directory = convergence.trial_directory(scratch, expected)
    directory.mkdir()
    store = convergence.TrialStore(directory, expected)
    features.seed_everything(config.seed)
    head = features.make_head("linear", caches["train"], 2, 4)
    run = convergence.ConvergenceRun(head, caches, config, store)
    run.block(initial=True)
    run.block()
    state = copy.deepcopy(store.load())
    assert len(state["plateau"]["recent_losses"]) == 2
    run.store = None
    run.block()
    uninterrupted = copy.deepcopy(run.state())
    (directory / "curve.csv").write_text("broken")
    (directory / "progress.json").write_text("{")
    resumed = convergence.ConvergenceRun(
        features.make_head("linear", caches["train"], 2, 4), caches, config, store
    )
    resumed.restore(store.load())
    assert_trees_equal(resumed.optimizer.state_dict(), state["optimizer"])
    assert resumed.plateau == convergence.Plateau(**state["plateau"])
    resumed.block()
    for name in ("model", "optimizer", "rng", "counters", "plateau"):
        assert_trees_equal(resumed.state()[name], uninterrupted[name])
    with (directory / "curve.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(resumed.curve)
    assert (
        json.loads((directory / "progress.json").read_text())["block"] == resumed.counters["blocks"]
    )


def test_immutable_results_atomic_aggregation_and_resume_failures(
    payload: dict, config: convergence.TrialConfig, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caches, metadata = selected(payload, ["deep"])
    job = study.Job(config, "deep", ("deep",))
    result = convergence.run_trial(config, caches, metadata, scratch / "trials")
    expected = convergence.scientific_identity(config, metadata)
    directory = convergence.trial_directory(scratch / "trials", expected)
    original = (directory / "result.json").read_bytes()
    with pytest.raises(FileExistsError):
        convergence.run_trial(config, caches, metadata, scratch / "trials")
    monkeypatch.setattr(
        features, "make_head", Mock(side_effect=AssertionError("completed means immutable"))
    )
    assert convergence.run_trial(
        config, caches, metadata, scratch / "trials", resume=True
    ) == json.loads(json.dumps(result))
    assert (directory / "result.json").read_bytes() == original
    pending = study.Job(replace(config, lr=0.1), "deep", ("deep",))
    study.aggregate(scratch, [job, pending], payload["metadata"])
    with (scratch / "combined.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["not_converged", "incomplete"]
    assert rows[0]["parameter_count"] == str(result["parameter_count"])
    with (scratch / "validation_selected.csv").open() as stream:
        assert next(csv.DictReader(stream))["selection_status"] == "incomplete_grid"
    assert not list(scratch.rglob("*.writing"))
    corrupted = json.loads(original)
    corrupted["test"]["miou"] = 0.12345
    features.atomic_json(directory / "result.json", corrupted)
    with pytest.raises(ValueError, match="checksum"):
        study.aggregate(scratch, [job], payload["metadata"])
    assert (scratch / "combined.csv").read_text().count("incomplete") == 1


def test_validation_lr_selection_uses_seed_mean_never_test() -> None:
    rows = []
    for lr, scores, test in [(0.01, [0.9, 0.1], 0.99), (0.1, [0.6, 0.6], 0.1)]:
        for seed, score in enumerate(scores):
            rows.append(
                {
                    "dataset": "x",
                    "model": "y",
                    "head": "linear",
                    "layer_group": "deep",
                    "optimizer": "adam",
                    "layers": '["deep"]',
                    "layer_count": 1,
                    "status": "not_converged",
                    "lr": lr,
                    "seed": seed,
                    "best_val_miou": score,
                    "test_miou": test,
                    "parameter_count": 10,
                    "optimization_seconds": 3,
                    "training_wall_seconds": 6,
                    "selected_optimization_seconds": 1,
                    "stationary": False,
                    "mixed_hardware_timings": False,
                    "hardware_history": "[]",
                }
            )
    summary = study.validation_summary(rows)[0]
    assert summary["selected_lr"] == 0.1
    assert summary["mean_test_miou"] == 0.1
    assert summary["converged_seeds"] == 0
    assert summary["selection_status"] == "validation_selected"
    rows[0]["status"] = "failed"
    assert study.validation_summary(rows)[0]["selection_status"] == "incomplete_grid"


@pytest.mark.parametrize("failure", ["nonzero", "exception", "missing"])
def test_scheduler_propagates_failures_never_fake_completed(
    failure: str,
    payload: dict,
    config: convergence.TrialConfig,
    scratch: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = study.Job(config, "deep", ("deep",))
    runner = runner_for(scratch, payload, [job])
    if failure == "exception":
        monkeypatch.setattr(runner, "_run_job", Mock(side_effect=OSError("spawn error")))
        expected_error = OSError
    else:
        monkeypatch.setattr(runner, "_run_attempt", lambda *_args: failure == "missing")
        expected_error = RuntimeError
    with pytest.raises(expected_error):
        runner._dispatch([job])
    assert runner.counts["failed"] == 1
    assert runner.counts["completed"] == 0
    assert runner.counts["running"] == 0


def test_scheduler_finishes_adam_before_lbfgs(
    payload: dict, config: convergence.TrialConfig, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    jobs = [
        study.Job(replace(config, optimizer=optimizer, seed=seed), "deep", ("deep",))
        for optimizer in ("adam", "lbfgs")
        for seed in (0, 1)
    ]
    runner = runner_for(scratch, payload, jobs)
    done = set()

    def execute(job: study.Job, _gpu: int) -> bool:
        if job.trial.optimizer == "lbfgs":
            assert all(first.job_id in done for first in jobs[:2])
        done.add(job.job_id)
        return True

    monkeypatch.setattr(runner, "_is_complete", lambda job: job.job_id in done)
    monkeypatch.setattr(runner, "_run_job", execute)
    runner.run()
    assert runner.counts["completed"] == 4


def test_numerical_failure_is_immutable_but_not_success(
    payload: dict, config: convergence.TrialConfig, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caches, metadata = selected(payload, ["deep"])
    monkeypatch.setattr(
        convergence.FullObjective,
        "monitor",
        Mock(side_effect=features.DivergedError("bad objective")),
    )
    result = convergence.run_trial(config, caches, metadata, scratch / "trials")
    assert result["status"] == "failed"
    assert result["stationary"] is False
    assert "test" not in result
    job = study.Job(config, "deep", ("deep",))
    runner = runner_for(scratch, payload, [job])
    with pytest.raises(RuntimeError, match="Immutable failed trial"):
        runner._is_complete(job)
    study.aggregate(scratch, [job], payload["metadata"])
    with (scratch / "combined.csv").open() as stream:
        assert next(csv.DictReader(stream))["status"] == "failed"


def test_prepare_cache_encodes_complete_splits_once_through_benchmodel(
    scratch: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Backbone(features.BenchModel):
        def __init__(self) -> None:
            nn.Module.__init__(self)
            self.backbone = nn.ModuleDict({name: nn.Identity() for name in ("deep", "shallow")})
            self.calls = 0

        def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
            return images * 2

        def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            for module in self.backbone.values():
                images = module(images)
            return images.mean((2, 3))

    spec = {
        **payload["metadata"]["spec"],
        "layers": ["deep", "shallow"],
        "bands": "all",
        "extraction_seed": 0,
        "extract_batch_size": 2,
        "interpolation": "bilinear",
        "model_config": {},
    }
    loaders = [
        DataLoader(
            [
                {"image": torch.ones(3, 8, 8), "mask": torch.zeros(8, 8, dtype=torch.long)}
                for _ in range(count)
            ],
            batch_size=2,
        )
        for count in (5, 3, 2)
    ]
    backbone = Backbone()
    instantiate = Mock(return_value=backbone)
    monkeypatch.setattr(features, "instantiate", instantiate)
    monkeypatch.setattr(features, "get_datasets", lambda **_kwargs: (None, *loaders))
    monkeypatch.setattr(
        features, "get_bench_dataset_class", lambda _name: lambda: SimpleNamespace(bands=[])
    )
    path = scratch / "features.pt"
    first = features.prepare_cache(path, spec, CPU, 0)
    second = features.prepare_cache(path, spec, CPU, 0)
    assert backbone.calls == 6
    assert instantiate.call_count == 1
    assert first["metadata"]["tensor_sha256"] == second["metadata"]["tensor_sha256"]
    assert first["metadata"]["feature_shapes"] == [[3, 8, 8], [3, 8, 8]]
    assert first["splits"]["train"]["features"][0].eq(2).all()
    assert all(not module._forward_hooks for module in backbone.modules())


def test_worker_subprocess_executes_manifest_and_resumes(
    payload: dict, config: convergence.TrialConfig, scratch: Path
) -> None:
    spec = payload["metadata"]["spec"]
    spec["versions"] = features.software_versions()
    payload["metadata"]["key"] = features.identity(spec)
    cache = scratch / "features.pt"
    manifest = scratch / "job.json"
    features.atomic_save(cache, payload)
    job = study.Job(replace(config, max_iterations=1), "deep", ("deep",))
    features.atomic_json(
        manifest,
        {
            "job": study.asdict(job),
            "cache_spec": spec,
            "cache_tensor_sha256": payload["metadata"]["tensor_sha256"],
        },
    )
    command = [
        sys.executable,
        str(ROOT / "scripts/run_segmentation_layer_study.py"),
        "--worker",
        str(manifest),
        "--device",
        "cpu",
        "--cache",
        str(cache),
        "--output-dir",
        str(scratch),
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    directory = convergence.trial_directory(
        scratch / "trials", study.job_identity(job, payload["metadata"])
    )
    before = (directory / "result.json").read_bytes()
    subprocess.run([*command, "--resume"], cwd=ROOT, check=True, capture_output=True, text=True)
    assert (directory / "result.json").read_bytes() == before
    result = convergence.completed_result(directory, study.job_identity(job, payload["metadata"]))
    assert result["stop_reason"] == "safety_max_iterations"
    assert result["status"] == "not_converged"


def test_hardware_changes_are_labeled_not_part_of_scientific_identity(
    payload: dict, config: convergence.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    caches, metadata = selected(payload)
    expected = convergence.scientific_identity(config, metadata)
    head = features.make_head("linear", caches["train"], 2, 4)
    monkeypatch.setattr(features, "hardware", lambda _device: {"device": "cuda:0", "name": "A"})
    first = convergence.ConvergenceRun(head, caches, config)
    state = copy.deepcopy(first.state())
    monkeypatch.setattr(features, "hardware", lambda _device: {"device": "cuda:3", "name": "A"})
    moved = convergence.ConvergenceRun(head, caches, config)
    moved.restore(copy.deepcopy(state))
    assert convergence.scientific_identity(config, metadata) == expected
    assert moved.summary()["mixed_hardware_timings"] is False
    assert len(moved.hardware_history) == 2
    monkeypatch.setattr(features, "hardware", lambda _device: {"device": "cuda:1", "name": "B"})
    changed = convergence.ConvergenceRun(head, caches, config)
    changed.restore(state)
    assert changed.summary()["mixed_hardware_timings"] is True


def test_single_vs_four_capacity_is_measured_not_assumed(payload: dict) -> None:
    four, _ = selected(payload)
    one, _ = selected(payload, ["deep"])
    one_head = features.make_head("linear", one["train"], 2, 4)
    four_head = features.make_head("linear", four["train"], 2, 4)
    assert sum(p.numel() for p in one_head.parameters()) < sum(
        p.numel() for p in four_head.parameters()
    )


def test_study_resume_refuses_config_changes_but_allows_gpu_rescheduling(
    scratch: Path, payload: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 6)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(features, "cache_spec", lambda *_args: payload["metadata"]["spec"])
    prepare = Mock(return_value=payload)
    monkeypatch.setattr(features, "prepare_cache", prepare)
    run = Mock()
    monkeypatch.setattr(study.StudyRunner, "run", run)
    arguments = [
        "--heads",
        "linear",
        "--seeds",
        "0",
        "--adam-lrs",
        "0.001",
        "--phase",
        "adam",
        "--output-dir",
        str(scratch),
    ]
    study.main("optimizer", "", [*arguments, "--gpus", "0"])
    with pytest.raises(FileExistsError, match="use --resume"):
        study.main("optimizer", "", arguments)
    study.main("optimizer", "", [*arguments, "--resume", "--gpus", "3,5"])
    assert run.call_count == 2
    assert prepare.call_count == 2
    before = (scratch / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="Incompatible study configuration"):
        study.main("optimizer", "", [*arguments, "--resume", "--batch-size", "3"])
    assert prepare.call_count == 2
    assert (scratch / "metadata.json").read_bytes() == before


def test_atomic_write_without_directory_fsync_flag(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = scratch / "portable.json"
    path.write_text("old")
    fsync = Mock(wraps=features.os.fsync)
    directory_open = Mock(side_effect=AssertionError("Windows cannot open directories for fsync"))
    with monkeypatch.context() as context:
        context.delattr(features.os, "O_DIRECTORY", raising=False)
        context.setattr(features.os, "fsync", fsync)
        context.setattr(features.os, "open", directory_open)
        features.atomic_json(path, {"committed": True})
        assert fsync.call_count == 1
        directory_open.assert_not_called()
    assert json.loads(path.read_text()) == {"committed": True}
    assert not path.with_name(path.name + ".writing").exists()

    def interrupted_write() -> None:
        with features.atomic_stream(path, "w") as stream:
            stream.write("incomplete")
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        interrupted_write()
    assert json.loads(path.read_text()) == {"committed": True}
    assert not path.with_name(path.name + ".writing").exists()


@pytest.mark.parametrize("suite", ["optimizer", "layer"])
def test_missing_dpt_dependency_fails_before_cache_or_workers(
    suite: str, scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importer = Mock(side_effect=ModuleNotFoundError("missing", name="transformers"))
    prepare = Mock(side_effect=AssertionError("must not extract"))
    launch = Mock(side_effect=AssertionError("must not schedule"))
    monkeypatch.setattr(features, "import_module", importer)
    monkeypatch.setattr(features, "prepare_cache", prepare)
    monkeypatch.setattr(study.StudyRunner, "run", launch)
    monkeypatch.setattr(
        torch.cuda, "device_count", Mock(side_effect=AssertionError("preflight first"))
    )
    output = scratch / "not-created"
    arguments = ["--heads", "dpt", "--output-dir", str(output)]
    if suite == "layer":
        arguments += ["--layer-groups", "spread_four"]
    with pytest.raises(ModuleNotFoundError, match="optional transformers dependency"):
        study.main(suite, "", arguments)
    importer.assert_called_once_with("transformers.models.dpt.modeling_dpt")
    prepare.assert_not_called()
    launch.assert_not_called()
    assert not output.exists()


def test_worker_checks_dpt_dependency_before_loading_cache(
    scratch: Path, payload: dict, config: convergence.TrialConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = study.Job(
        replace(config, head="dpt"), "four", tuple(payload["metadata"]["spec"]["layers"])
    )
    path = scratch / "job.json"
    features.atomic_json(
        path, {"job": study.asdict(job), "cache_spec": payload["metadata"]["spec"]}
    )
    monkeypatch.setattr(
        features,
        "import_module",
        Mock(side_effect=ModuleNotFoundError("missing", name="transformers")),
    )
    load = Mock(side_effect=AssertionError("preflight before tensor load"))
    monkeypatch.setattr(features, "load_cache", load)
    with pytest.raises(ModuleNotFoundError, match="optional transformers dependency"):
        study.run_worker(SimpleNamespace(worker=path))
    load.assert_not_called()


def test_dpt_dry_run_does_not_import_optional_dependency(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    importer = Mock(side_effect=AssertionError("dry runs must not import transformers"))
    monkeypatch.setattr(features, "import_module", importer)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    output = scratch / "not-created"
    study.main(
        "optimizer",
        "",
        [
            "--heads",
            "dpt",
            "--seeds",
            "0",
            "--adam-lrs",
            "0.001",
            "--lbfgs-lrs",
            "1",
            "--dry-run",
            "--output-dir",
            str(output),
        ],
    )
    importer.assert_not_called()
    assert not output.exists()


def test_preflight_checks_decoder_api_but_does_not_import_for_other_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    importer = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(features, "import_module", importer)
    features.preflight_heads(["linear", "conv_block", "fpn", "patch_linear"])
    importer.assert_not_called()
    with pytest.raises(ImportError, match="lacks the DPTFeatureFusionLayer"):
        features.preflight_heads(["dpt"])


def test_recent_plateau_window_starts_at_minimum_iterations(
    config: convergence.TrialConfig,
) -> None:
    config = replace(config, min_iterations=4, patience=3)
    plateau = convergence.Plateau()
    for iteration in range(4):
        assert plateau.observe(0.5, iteration, config) is None
    assert plateau.recent_losses == []
    assert plateau.observe(0.5, 4, config) is None
    assert plateau.observe(0.5, 5, config) is None
    assert plateau.observe(0.5, 6, config) == "train_loss_plateau"
    assert plateau.recent_losses == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("patience", [2, 3, 10])
def test_recovery_above_early_best_never_stops_without_explicit_safety_cap(
    patience: int, payload: dict, config: convergence.TrialConfig
) -> None:
    config = replace(config, patience=patience, min_iterations=0, max_iterations=None)
    caches, _ = selected(payload)
    run = convergence.ConvergenceRun(
        features.make_head("linear", caches["train"], 2, 4), caches, config
    )
    losses = (
        [0.5, 35.7, 5.8, 2.91]
        if patience < 10
        else [
            0.5,
            35.7,
            30,
            25,
            20,
            16,
            12,
            9,
            7,
            5.8,
            4.5,
            3.8,
            2.91,
        ]
    )
    for iteration, loss in enumerate(losses):
        run.counters["epochs"] = iteration
        assert run.stopping_reason(loss, None, changed=True) is None
    assert run.plateau.stale_checks >= config.patience
    assert run.plateau.reference == 0.5
    assert (
        run.plateau.window_statistics(config)["recent_ce_half_mean_decrease"] > config.absolute_tol
    )


def test_recent_range_accumulates_small_recovery_steps(config: convergence.TrialConfig) -> None:
    config = replace(config, patience=5, min_iterations=0, absolute_tol=0.01, relative_tol=0)
    plateau = convergence.Plateau()
    assert plateau.observe(0.5, 0, config) is None
    for iteration in range(1, 15):
        assert plateau.observe(1.0 - 0.004 * iteration, iteration, config) is None
    window = plateau.window_statistics(config)
    assert window["recent_ce_range"] > window["recent_ce_tolerance"]
    assert window["recent_ce_half_mean_decrease"] > window["recent_ce_tolerance"]
    restored = convergence.Plateau(**study.asdict(plateau))
    assert restored.recent_losses == plateau.recent_losses
    assert restored.observe(0.94, 15, config) == plateau.observe(0.94, 15, config)


@pytest.mark.parametrize(
    ("losses", "outcome"),
    [
        ([0.5, 5.0, 3.0, 5.0], ("no_best_improvement", "not_converged")),
        ([0.5, 1.0, 2.0, 3.0], ("no_best_improvement", "not_converged")),
        ([0.5, 0.5, 0.5, 0.5], ("train_loss_plateau", "converged")),
        ([0.5, 35.7, 5.8, 2.91], (None, "running")),
    ],
)
def test_recent_window_stop_reasons_never_claim_stationarity(
    losses: list[float],
    outcome: tuple[str | None, str],
    payload: dict,
    config: convergence.TrialConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(config, patience=3, min_iterations=0, max_iterations=None)
    caches, _ = selected(payload)
    run = convergence.ConvergenceRun(
        features.make_head("linear", caches["train"], 2, 4), caches, config
    )
    stream = iter(losses)
    monkeypatch.setattr(run.objective, "monitor", lambda **_kwargs: (next(stream), None))
    for iteration in range(len(losses)):
        run.counters["epochs"] = iteration
        assert run.status == "running"
        run.block(initial=True)
    assert run.status == outcome[1]
    assert run.stop_reason == outcome[0]
    assert run.summary()["stationary"] is False
    assert run.curve[-1]["recent_ce_range"] == max(losses[-3:]) - min(losses[-3:])


def test_recent_flatness_uses_relative_tolerance(config: convergence.TrialConfig) -> None:
    config = replace(config, patience=3, min_iterations=0, absolute_tol=0.01, relative_tol=0.001)
    plateau = convergence.Plateau()
    for iteration, loss in enumerate([1000.0, 1000.1, 1000.2]):
        assert plateau.observe(loss, iteration, config) is None
    assert plateau.observe(1000.3, 3, config) == "train_loss_plateau"
    assert plateau.window_statistics(config)["recent_ce_tolerance"] > 1.0
