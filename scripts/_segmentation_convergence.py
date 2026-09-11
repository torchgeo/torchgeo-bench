"""Resumable accepted-training-CE convergence, shared by both production studies.

Adam is constant-rate with zero decay. L-BFGS keeps strong-Wolfe history across
20-iteration chunks; closures accumulate globally valid-pixel-weighted gradients
over fixed microbatches. Adam shuffles these same batches, never their membership,
and scales summed batch CE by the number of batches / total valid pixels.
Train-mode BN makes this shared objective dependent on fixed batch membership.
Convergence requires a flat recent loss window after minimum iterations, not just
stale historical best loss. Stale, nonrecovering oscillations are not_converged;
materially falling windows continue, even above the historical best. Plateaus are
operational convergence, not stationarity. Validation selects the earliest
strictly best checkpoint; test never controls training or selection.

Each block commits model/optimizer/RNG/buffers/patience/curve to checkpoint.pt
before publishing derived CSV/progress. Resumption replays only uncommitted work.
Active training wall time excludes offline time. Optimization timing excludes
monitoring, calibration, validation, selection, persistence and final evaluation.
"""

import csv
import json
import logging
import math
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import _segmentation_features as shared
import numpy as np
import torch
from filelock import FileLock
from torch import nn

logger = logging.getLogger(__name__)
STAGES = (
    "setup",
    "warmup",
    "optimization",
    "convergence",
    "calibration",
    "validation",
    "selection",
    "checkpoint",
    "test",
    "bootstrap",
)


@dataclass(frozen=True)
class TrialConfig:
    """Scientific identity of a constant-LR, unregularized convergence fit."""

    head: str
    optimizer: str
    lr: float
    seed: int = 0
    hidden_dim: int = 256
    batch_size: int = 8
    check_every: int = 5
    lbfgs_iterations: int = 20
    history_size: int = 10
    patience: int = 10
    min_iterations: int = 50
    absolute_tol: float = 1e-5
    relative_tol: float = 1e-4
    gradient_tol: float = 1e-7
    numerical_patience: int = 3
    max_iterations: int | None = None
    bootstrap: int = 1000

    def __post_init__(self) -> None:
        if self.head not in shared.HEADS or self.optimizer not in ("adam", "lbfgs"):
            raise ValueError("Unknown head or optimizer.")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite.")
        if any(
            not math.isfinite(v) or v < 0
            for v in (self.absolute_tol, self.relative_tol, self.gradient_tol)
        ):
            raise ValueError("Tolerances must be finite and nonnegative.")
        counts = (
            self.hidden_dim,
            self.batch_size,
            self.check_every,
            self.lbfgs_iterations,
            self.history_size,
            self.patience,
            self.numerical_patience,
            self.bootstrap,
        )
        if min(counts) < 1 or self.min_iterations < 0 or not 0 <= self.seed < 2**32:
            raise ValueError("Counts must be positive and seed in [0, 2**32).")
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations is a positive optional safety cap.")


@dataclass
class Plateau:
    """Track cumulative best improvement and recent, post-minimum loss behavior."""

    reference: float | None = None
    best_loss: float | None = None
    stale_checks: int = 0
    checks: int = 0
    recent_losses: list[float] = field(default_factory=list)

    def window_statistics(self, config: TrialConfig) -> dict[str, float | None]:
        """Measure range and equal-half mean decrease, excluding an odd middle value."""
        if not self.recent_losses:
            return {
                "recent_ce_range": None,
                "recent_ce_half_mean_decrease": None,
                "recent_ce_tolerance": None,
            }
        half = len(self.recent_losses) // 2
        decrease = (
            (math.fsum(self.recent_losses[:half]) - math.fsum(self.recent_losses[-half:])) / half
            if half
            else None
        )
        return {
            "recent_ce_range": max(self.recent_losses) - min(self.recent_losses),
            "recent_ce_half_mean_decrease": decrease,
            "recent_ce_tolerance": max(
                config.absolute_tol, config.relative_tol * abs(min(self.recent_losses))
            ),
        }

    def observe(self, loss: float, iterations: int, config: TrialConfig) -> str | None:
        """Distinguish a flat window from stale, nonrecovering historical-best patience.

        Small improvements accumulate against ``reference``. Only checks at or
        after ``min_iterations`` enter the recent window, which holds ``patience``
        observations (at least two to measure a trend). Stale patience alone must
        never stop a materially recovering window above an early historical best.
        """
        if not math.isfinite(loss):
            raise shared.DivergedError("Nonfinite accepted training loss.")
        self.checks += 1
        self.best_loss = loss if self.best_loss is None else min(self.best_loss, loss)
        threshold = max(config.absolute_tol, config.relative_tol * abs(self.reference or 0))
        if self.reference is None or self.reference - loss > threshold:
            self.reference, self.stale_checks = loss, 0
        else:
            self.stale_checks += 1
        if iterations < config.min_iterations:
            return None
        window_size = max(2, config.patience)
        self.recent_losses.append(loss)
        del self.recent_losses[:-window_size]
        if len(self.recent_losses) < window_size or self.stale_checks < config.patience:
            return None
        window = self.window_statistics(config)
        if window["recent_ce_range"] <= window["recent_ce_tolerance"]:
            return "train_loss_plateau"
        if window["recent_ce_half_mean_decrease"] <= window["recent_ce_tolerance"]:
            return "no_best_improvement"
        return None


class Timings:
    """Synchronized cumulative stage costs, including stages that fail."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.seconds = dict.fromkeys(STAGES, 0.0)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Accumulate wall time after synchronizing device work."""
        shared.synchronize(self.device)
        start = time.perf_counter()
        try:
            yield
        finally:
            shared.synchronize(self.device)
            self.seconds[stage] += time.perf_counter() - start


@contextmanager
def preserved_training_buffers(head: nn.Module) -> Iterator[None]:
    """Use train-mode statistics without leaking buffers or module modes."""
    buffers = [(buffer, buffer.detach().clone()) for buffer in head.buffers()]
    modes = [(module, module.training) for module in head.modules()]
    head.train()
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, original in buffers:
                buffer.copy_(original)
        for module, training in modes:
            module.training = training


def gradient_norm(head: nn.Module) -> float:
    """Return accepted-gradient infinity norm with a single host reduction."""
    gradients = [p.grad.detach().abs().amax() for p in head.parameters() if p.grad is not None]
    value = float(torch.stack(gradients).amax()) if gradients else 0.0
    if not math.isfinite(value):
        raise shared.DivergedError("Nonfinite accepted gradient.")
    return value


class FullObjective:
    """Resident, fixed-order global valid-pixel CE with memory-bounded autograd."""

    def __init__(self, head: nn.Module, cache: shared.GPUTensorCache, batch_size: int) -> None:
        shared.reject_stochastic_modules(head)
        self.head, self.cache, self.batch_size = head, cache, batch_size
        self.valid_per_image = cache.masks.ne(shared.IGNORE_INDEX).sum((1, 2)).cpu()
        self.valid_pixels = int(self.valid_per_image.sum())
        if not self.valid_pixels:
            raise ValueError("Training split has no valid pixels.")
        self.evaluations = 0
        self.monitor_evaluations = 0
        self.batch_count = (len(cache) + batch_size - 1) // batch_size

    def compute(self, *, backward: bool) -> torch.Tensor:
        """Compute one microbatch graph at a time, preserving train-mode buffers."""
        if backward:
            self.head.zero_grad(set_to_none=True)
        total = torch.zeros((), device=self.cache.device)
        with preserved_training_buffers(self.head), torch.set_grad_enabled(backward):
            for features, masks in self.cache.ordered_batches(self.batch_size):
                loss = shared.summed_cross_entropy(self.head(features, *masks.shape[-2:]), masks)
                loss = loss / self.valid_pixels
                if backward:
                    loss.backward()
                total += loss.detach()
        if not torch.isfinite(total):
            raise shared.DivergedError("Nonfinite full training objective.")
        return total

    def __call__(self) -> torch.Tensor:
        """Supply the full backward closure while counting actual evaluations."""
        self.evaluations += 1
        return self.compute(backward=True)

    def monitor(self, *, backward: bool) -> tuple[float, float | None]:
        """Recompute accepted weights, separately from optimization timing."""
        self.monitor_evaluations += 1
        loss = self.compute(backward=backward)
        return float(loss), gradient_norm(self.head) if backward else None


def adam_epoch(objective: FullObjective, optimizer: torch.optim.Adam, counters: dict) -> None:
    """Shuffle fixed BN buckets with uniform-batch unbiased global-CE scaling."""
    head, cache = objective.head, objective.cache
    head.train()
    total = torch.zeros((), device=cache.device)
    for batch_index in torch.randperm(objective.batch_count).tolist():
        start = batch_index * objective.batch_size
        stop = min(start + objective.batch_size, len(cache))
        bucket = slice(start, stop)
        counters["samples"] += stop - start
        if not objective.valid_per_image[bucket].sum():
            continue
        features = [tensor[bucket] for tensor in cache.layer_tensors]
        masks = cache.masks[bucket]
        optimizer.zero_grad(set_to_none=True)
        loss = shared.summed_cross_entropy(head(features, *masks.shape[-2:]), masks)
        loss = loss * objective.batch_count / objective.valid_pixels
        loss.backward()
        optimizer.step()
        total += loss.detach()
        counters["optimizer_step_calls"] += 1
        counters["optimizer_updates"] += 1
    if not torch.isfinite(total):
        raise shared.DivergedError("Nonfinite Adam epoch.")
    counters["epochs"] += 1


def optimizer_settings(config: TrialConfig) -> dict:
    """Record every explicitly selected optimizer option."""
    if config.optimizer == "adam":
        return {
            "lr": config.lr,
            "weight_decay": 0.0,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "amsgrad": False,
            "foreach": True,
            "fused": False,
        }
    return {
        "lr": config.lr,
        "max_iter": config.lbfgs_iterations,
        "max_eval": max(25, config.lbfgs_iterations * 5 // 4),
        "history_size": config.history_size,
        "line_search_fn": "strong_wolfe",
        "tolerance_grad": config.gradient_tol,
        "tolerance_change": 1e-9,
    }


def model_snapshot(head: nn.Module) -> dict:
    """Capture parameters, nonpersistent buffers and individual module flags."""
    return {
        "state_dict": shared.snapshot(head),
        "buffers": {name: buffer.detach().cpu().clone() for name, buffer in head.named_buffers()},
        "modes": {name: module.training for name, module in head.named_modules()},
    }


def restore_model(head: nn.Module, state: dict) -> None:
    """Restore all state, including nonpersistent buffers."""
    head.load_state_dict(state["state_dict"])
    with torch.no_grad():
        for name, buffer in head.named_buffers():
            buffer.copy_(state["buffers"][name])
    for name, module in head.named_modules():
        module.training = state["modes"][name]


def rng_snapshot(device: torch.device) -> dict:
    """Serialize all used RNGs as safe primitive values and tensors."""
    numpy_state = np.random.get_state()  # noqa: NPY002 -- Preserve upstream global RNG.
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state: dict, device: torch.device) -> None:
    """Resume permutations and RNG streams, allowing a different CUDA index."""
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(  # noqa: NPY002 -- Restore upstream global RNG.
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def warmup(head: nn.Module, cache: shared.GPUTensorCache, batch_size: int) -> None:
    """Warm dummy forward/backward without changing model state or any RNG."""
    state, rng = model_snapshot(head), rng_snapshot(torch.device(cache.device))
    try:
        head.train()
        count = min(batch_size, len(cache))
        features = [torch.zeros_like(t[:count]) for t in cache.layer_tensors]
        masks = torch.zeros_like(cache.masks[:count])
        (
            shared.summed_cross_entropy(head(features, *masks.shape[-2:]), masks) / masks.numel()
        ).backward()
    finally:
        restore_model(head, state)
        head.zero_grad(set_to_none=True)
        restore_rng(rng, torch.device(cache.device))


class TrialStore:
    """Treat checkpoint.pt as the authoritative commit, repairing derived files."""

    def __init__(self, directory: Path, trial_identity: dict) -> None:
        self.directory, self.identity = directory, trial_identity

    def load(self) -> dict:
        """Validate identity and recover accounting from a complete same-block progress file."""
        state = torch.load(self.directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        if state["identity"] != self.identity:
            raise ValueError("Checkpoint identity mismatch.")
        path = self.directory / "progress.json"
        if path.exists():
            try:
                progress = json.loads(path.read_text())
            except json.JSONDecodeError:  # allow-except: the checkpoint is authoritative
                logger.warning(
                    "Invalid progress JSON at %s; restoring timing from checkpoint.", path
                )
                progress = None
            if progress is not None:
                if progress["identity"] != self.identity:
                    raise ValueError("Progress identity mismatch.")
                if progress["block"] == state["counters"]["blocks"]:
                    state["timings_seconds"] = progress["timings_seconds"]
                    state["training_wall_seconds"] = progress["training_wall_seconds"]
        return state

    def write_checkpoint(self, run: "ConvergenceRun") -> None:
        """Commit continuation before derived curve, so crashes cannot advance only the CSV."""
        with run.timings.measure("checkpoint"):
            shared.atomic_save(
                self.directory / "checkpoint.pt", {"identity": self.identity, **run.state()}
            )
            with shared.atomic_stream(self.directory / "curve.csv", "w") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(run.curve[0]) if run.curve else ["block"]
                )
                writer.writeheader()
                writer.writerows(run.curve)

    def progress(self, run: "ConvergenceRun") -> None:
        """Publish complete block accounting and live scientific status."""
        shared.atomic_json(
            self.directory / "progress.json",
            {
                "identity": self.identity,
                "status": run.status,
                "stop_reason": run.stop_reason,
                "block": run.counters["blocks"],
                "latest": run.curve[-1] if run.curve else None,
                "timings_seconds": run.timings.seconds,
                "training_wall_seconds": run.training_wall_seconds,
                "plateau": asdict(run.plateau),
            },
        )


class ConvergenceRun:
    """Resumable training state machine with validation-only checkpoint selection."""

    def __init__(
        self, head: nn.Module, caches: dict, config: TrialConfig, store: TrialStore | None = None
    ) -> None:
        self.head, self.caches, self.config, self.store = head, caches, config, store
        self.device = torch.device(caches["train"].device)
        self.timings = Timings(self.device)
        with self.timings.measure("setup"):
            self.objective = FullObjective(head, caches["train"], config.batch_size)
            constructor = torch.optim.Adam if config.optimizer == "adam" else torch.optim.LBFGS
            self.optimizer = constructor(head.parameters(), **optimizer_settings(config))
            self.initial_state_sha256 = shared.tensor_digest(head.state_dict())
        self.best: dict | None = None
        self.plateau = Plateau()
        self.curve: list[dict] = []
        self.counters = dict.fromkeys(
            (
                "blocks",
                "epochs",
                "optimizer_step_calls",
                "optimizer_updates",
                "samples",
                "calibration_passes",
                "unchanged_blocks",
                "stalled_iteration_blocks",
            ),
            0,
        )
        self.status, self.stop_reason = "running", None
        self.training_wall_seconds = 0.0
        self.failure: str | None = None
        self.hardware_history = [shared.hardware(self.device)]

    @property
    def lbfgs_state(self) -> dict:
        """Return actual persistent PyTorch iteration/history/evaluation counters."""
        return self.optimizer.state.get(next(self.head.parameters()), {})

    @property
    def iterations(self) -> int:
        """Count Adam epochs or actual L-BFGS internal iterations, not closure calls."""
        return (
            self.counters["epochs"]
            if self.config.optimizer == "adam"
            else int(self.lbfgs_state.get("n_iter", 0))
        )

    def state(self) -> dict:
        """Capture every value needed to continue a committed block exactly."""
        return {
            "model": model_snapshot(self.head),
            "optimizer": self.optimizer.state_dict(),
            "rng": rng_snapshot(self.device),
            "counters": self.counters,
            "plateau": asdict(self.plateau),
            "curve": self.curve,
            "best": self.best,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "failure": self.failure,
            "initial_state_sha256": self.initial_state_sha256,
            "timings_seconds": self.timings.seconds,
            "training_wall_seconds": self.training_wall_seconds,
            "closure_evaluations": self.objective.evaluations,
            "monitor_evaluations": self.objective.monitor_evaluations,
            "hardware_history": self.hardware_history,
        }

    def restore(self, state: dict) -> None:
        """Continue without resetting momentum, curvature, RNG or patience."""
        setup_seconds = self.timings.seconds["setup"]
        restore_model(self.head, state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.counters = state["counters"]
        self.plateau = Plateau(**state["plateau"])
        self.curve, self.best = state["curve"], state["best"]
        self.status, self.stop_reason, self.failure = (
            state["status"],
            state["stop_reason"],
            state["failure"],
        )
        self.initial_state_sha256 = state["initial_state_sha256"]
        self.timings.seconds = state["timings_seconds"]
        self.timings.seconds["setup"] += setup_seconds
        self.training_wall_seconds = state["training_wall_seconds"]
        self.objective.evaluations = state["closure_evaluations"]
        self.objective.monitor_evaluations = state["monitor_evaluations"]
        self.hardware_history = state["hardware_history"]
        current = shared.hardware(self.device)
        if current != self.hardware_history[-1]:
            self.hardware_history.append(current)
        restore_rng(state["rng"], self.device)

    def optimize(self) -> bool:
        """Execute a block and report exact accepted parameter change for L-BFGS."""
        config = self.config
        remaining = (
            None if config.max_iterations is None else config.max_iterations - self.iterations
        )
        if config.optimizer == "adam":
            with self.timings.measure("optimization"):
                for _ in range(
                    config.check_every if remaining is None else min(config.check_every, remaining)
                ):
                    adam_epoch(self.objective, self.optimizer, self.counters)
            return True
        chunk = (
            config.lbfgs_iterations
            if remaining is None
            else min(config.lbfgs_iterations, remaining)
        )
        self.optimizer.param_groups[0]["max_iter"] = chunk
        self.optimizer.param_groups[0]["max_eval"] = max(25, chunk * 5 // 4)
        with self.timings.measure("convergence"):
            before = [p.detach().clone() for p in self.head.parameters()]
        previous_iterations = self.iterations
        with self.timings.measure("optimization"):
            self.optimizer.step(self.objective)
        with self.timings.measure("convergence"):
            changed = not bool(
                torch.stack(
                    [
                        torch.eq(p, original).all()
                        for p, original in zip(self.head.parameters(), before, strict=True)
                    ]
                ).all()
            )
        self.counters["optimizer_step_calls"] += 1
        self.counters["optimizer_updates"] += int(changed)
        stalled = self.iterations == previous_iterations
        self.counters["stalled_iteration_blocks"] = (
            self.counters["stalled_iteration_blocks"] + 1 if stalled else 0
        )
        return changed

    def stopping_reason(self, loss: float, gradient: float | None, *, changed: bool) -> str | None:
        """Separate stationarity, operational plateau, numerical stalls and safety caps."""
        config = self.config
        plateau_reason = self.plateau.observe(loss, self.iterations, config)
        if gradient is not None and gradient <= config.gradient_tol:
            return "gradient_tolerance"
        no_decrease = bool(self.curve) and loss >= self.curve[-1]["train_ce"]
        self.counters["unchanged_blocks"] = (
            self.counters["unchanged_blocks"] + 1 if not changed and no_decrease else 0
        )
        if (
            max(self.counters["unchanged_blocks"], self.counters["stalled_iteration_blocks"])
            >= config.numerical_patience
        ):
            return "numerical_stall"
        if plateau_reason is not None:
            return plateau_reason
        if config.max_iterations is not None and self.iterations >= config.max_iterations:
            return "safety_max_iterations"
        return None

    def row(self, loss: float, gradient: float | None, val: dict) -> dict:
        """Describe accepted weights with cumulative counters and stage costs."""
        return {
            "block": self.counters["blocks"],
            "iterations": self.iterations,
            "epochs": self.counters["epochs"],
            "train_ce": loss,
            "gradient_inf_norm": gradient,
            "val_ce": val["ce"],
            "val_miou": val["miou"],
            "plateau_reference_ce": self.plateau.reference,
            "stale_checks": self.plateau.stale_checks,
            **self.plateau.window_statistics(self.config),
            "optimizer_step_calls": self.counters["optimizer_step_calls"],
            "optimizer_updates": self.counters["optimizer_updates"],
            "lbfgs_n_iter": self.lbfgs_state.get("n_iter", 0),
            "lbfgs_func_evals": self.lbfgs_state.get("func_evals", 0),
            "lbfgs_history_length": len(self.lbfgs_state.get("old_dirs", [])),
            "closure_evaluations": self.objective.evaluations,
            "monitor_evaluations": self.objective.monitor_evaluations,
            "optimization_data_passes": self.objective.evaluations
            if self.config.optimizer == "lbfgs"
            else self.counters["samples"] / len(self.caches["train"]),
            "calibration_passes": self.counters["calibration_passes"],
            "status": self.status,
            "stop_reason": self.stop_reason,
            **{f"{stage}_seconds": value for stage, value in self.timings.seconds.items()},
        }

    def check(self, block_started_at: float, *, changed: bool) -> None:
        """Monitor training CE, calibrate BN on train, then select by validation mIoU."""
        with self.timings.measure("convergence"):
            loss, gradient = self.objective.monitor(backward=self.config.optimizer == "lbfgs")
            self.stop_reason = self.stopping_reason(loss, gradient, changed=changed)
            if self.stop_reason is not None:
                self.status = (
                    "converged"
                    if self.stop_reason in ("gradient_tolerance", "train_loss_plateau")
                    else "not_converged"
                )
        with self.timings.measure("calibration"):
            self.counters["calibration_passes"] += int(
                shared.recalibrate_bn(self.head, self.caches["train"], self.config.batch_size)
            )
        with self.timings.measure("validation"):
            val = shared.evaluate(self.head, self.caches["val"], self.config.batch_size)
        row = self.row(loss, gradient, val)
        selected = self.best is None or row["val_miou"] > self.best["val_miou"]
        with self.timings.measure("selection"):
            if selected:
                self.best = {**row, **model_snapshot(self.head)}
        row["selection_seconds"] = self.timings.seconds["selection"]
        row["training_wall_seconds"] = (
            self.training_wall_seconds + time.perf_counter() - block_started_at
        )
        if selected:
            self.best.update(
                selection_seconds=row["selection_seconds"],
                training_wall_seconds=row["training_wall_seconds"],
            )
        self.curve.append(row)

    def block(self, *, initial: bool = False) -> None:
        """Commit one monitored block; only numerical divergence becomes a failed result."""
        shared.synchronize(self.device)
        start = time.perf_counter()
        try:
            changed = True
            if not initial:
                changed = self.optimize()
                self.counters["blocks"] += 1
            self.check(start, changed=changed)
        except (
            shared.DivergedError
        ) as error:  # allow-except: record numerical failure as a trial result
            self.status, self.stop_reason, self.failure = "failed", "nonfinite", str(error)
        shared.synchronize(self.device)
        self.training_wall_seconds += time.perf_counter() - start
        if self.store:
            start = time.perf_counter()
            self.store.write_checkpoint(self)
            self.training_wall_seconds += time.perf_counter() - start
            self.store.progress(self)
        logger.info(
            "%s %s lr=%g seed=%d iterations=%d status=%s reason=%s optimization=%.3fs",
            self.config.head,
            self.config.optimizer,
            self.config.lr,
            self.config.seed,
            self.iterations,
            self.status,
            self.stop_reason,
            self.timings.seconds["optimization"],
        )

    def fit(self) -> dict:
        """Train until a declared stopping condition and retain both stopping/selected weights."""
        with self.timings.measure("warmup"):
            warmup(self.head, self.caches["train"], self.config.batch_size)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if not self.curve and self.status == "running":
            self.block(initial=True)
        while self.status == "running":
            self.block()
        return self.finish()

    def finish(self) -> dict:
        """Evaluate only validation-selected weights on test."""
        if self.store:
            with self.timings.measure("checkpoint"):
                shared.atomic_save(
                    self.store.directory / "final.pt",
                    {
                        "identity": self.store.identity,
                        **model_snapshot(self.head),
                        "counters": self.counters,
                    },
                )
        result = self.summary()
        if self.best is not None and self.status != "failed":
            with self.timings.measure("selection"):
                restore_model(self.head, self.best)
            with self.timings.measure("test"):
                test = shared.evaluate(self.head, self.caches["test"], self.config.batch_size)
            confusions = test.pop("per_image_confusions")
            with self.timings.measure("bootstrap"):
                result["test_miou_ci95"] = shared.bootstrap_interval(
                    confusions, self.config.bootstrap, self.config.seed
                )
            result["test"] = test
            if self.store:
                with self.timings.measure("checkpoint"):
                    shared.atomic_save(self.store.directory / "test_confusions.pt", confusions)
                    shared.atomic_save(
                        self.store.directory / "best.pt",
                        {"identity": self.store.identity, **self.best},
                    )
        result["timings_seconds"] = dict(self.timings.seconds)
        return result

    def summary(self) -> dict:
        """Report scientific outcome without conflating a plateau with stationarity."""
        hardware_types = {
            shared.identity({k: v for k, v in h.items() if k != "device"})
            for h in self.hardware_history
        }
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "stationary": self.stop_reason == "gradient_tolerance",
            "failure": self.failure,
            "initial_state_sha256": self.initial_state_sha256,
            "parameter_count": sum(p.numel() for p in self.head.parameters()),
            "trainable_parameter_count": sum(
                p.numel() for p in self.head.parameters() if p.requires_grad
            ),
            "optimizer_settings": optimizer_settings(self.config),
            "precision": "float32",
            "tf32": False,
            "weight_decay": 0.0,
            "training_wall_seconds": self.training_wall_seconds,
            "iterations": self.iterations,
            "counters": self.counters,
            "plateau": asdict(self.plateau),
            "final": self.curve[-1] if self.curve else None,
            "selected": {
                k: v for k, v in self.best.items() if k not in ("state_dict", "buffers", "modes")
            }
            if self.best
            else None,
            "closure_evaluations": self.objective.evaluations,
            "monitor_evaluations": self.objective.monitor_evaluations,
            "hardware_history": self.hardware_history,
            "mixed_hardware_timings": len(hardware_types) > 1,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
        }


def scientific_identity(config: TrialConfig, metadata: dict) -> dict:
    """Exclude GPU indices and hardware, but bind exact cache and selected connections."""
    return {
        "schema": 2,
        "trial": asdict(config),
        "cache_key": metadata["key"],
        "cache_tensor_sha256": metadata["tensor_sha256"],
        "selected_layers": metadata["selected_layers"],
        "selected_feature_shapes": metadata["selected_feature_shapes"],
        "versions": metadata["spec"]["versions"],
        "strict_determinism": metadata["spec"]["strict_determinism"],
    }


def trial_directory(output: Path, trial_identity: dict) -> Path:
    """Use full scientific digest, independent of scheduling and human labels."""
    return output / shared.identity(trial_identity)


def completed_result(directory: Path, trial_identity: dict) -> dict:
    """Load an immutable result after checking identity, integrity and required artifacts."""
    result = json.loads((directory / "result.json").read_text())
    checksum = result.pop("result_sha256")
    if shared.identity(result) != checksum:
        raise ValueError("Result checksum mismatch.")
    if result["identity"] != trial_identity:
        raise ValueError("Completed result identity mismatch.")
    required = ["identity.json", "checkpoint.pt", "curve.csv", "progress.json", "final.pt"]
    if result["status"] not in ("converged", "not_converged", "failed"):
        raise ValueError("Result does not have a terminal status.")
    if result["status"] != "failed":
        required.extend(["best.pt", "test_confusions.pt"])
        if not math.isfinite(result["test"]["miou"]) or not math.isfinite(
            result["selected"]["val_miou"]
        ):
            raise ValueError("Completed metrics must be finite.")
    if not all((directory / name).is_file() for name in required):
        raise ValueError(f"Incomplete completed trial in {directory}.")
    return result


def run_trial(
    config: TrialConfig, caches: dict, metadata: dict, output: Path, *, resume: bool = False
) -> dict:
    """Reserve and exactly continue one paired initialization, never overwrite a result."""
    device = torch.device(caches["train"].device)
    trial_identity = scientific_identity(config, metadata)
    directory = trial_directory(output, trial_identity)
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory) + ".lock", timeout=0):
        existing = directory.exists()
        if existing:
            if not resume:
                raise FileExistsError(f"Trial {directory} exists; use --resume.")
            if json.loads((directory / "identity.json").read_text()) != trial_identity:
                raise ValueError("Trial identity mismatch.")
            if (directory / "result.json").exists():
                return completed_result(directory, trial_identity)
        else:
            directory.mkdir()
            shared.atomic_json(directory / "identity.json", trial_identity)
        shared.synchronize(device)
        start = time.perf_counter()
        shared.seed_everything(config.seed)
        head = shared.make_head(
            config.head, caches["train"], metadata["spec"]["num_classes"], config.hidden_dim
        )
        shared.synchronize(device)
        setup_seconds = time.perf_counter() - start
        store = TrialStore(directory, trial_identity)
        run = ConvergenceRun(head, caches, config, store)
        run.timings.seconds["setup"] += setup_seconds
        if existing and (directory / "checkpoint.pt").exists():
            run.restore(store.load())
            store.write_checkpoint(run)
            store.progress(run)
        result = run.fit()
        result.update(
            identity=trial_identity,
            trial_key=shared.identity(trial_identity),
            cache_metadata=metadata,
        )
        shared.atomic_json(
            directory / "result.json", {**result, "result_sha256": shared.identity(result)}
        )
        return result
