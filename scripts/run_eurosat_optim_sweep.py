#!/usr/bin/env python
"""
Run linear probing experiments on m-eurosat using AdamW/SGD optimizer.

This script sweeps over learning rates and scheduler settings using a validation set,
then retrains the best configuration and evaluates on test (similar to how 
torchgeo_bench.py sweeps over C for logistic regression).

Usage:
    conda activate torchgeo
    python scripts/run_eurosat_optim_sweep.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.datasets import get_datasets
from src.models.interface import BenchModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for optimizer and training."""

    optimizer: str = "adamw"  # "adamw" or "sgd"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)  # AdamW only
    momentum: float = 0.9  # SGD only
    epochs: int = 200
    batch_size: int = 256
    patience: int = 50  # High patience for thorough search
    use_scheduler: bool = True


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    dataset_name: str = "m-eurosat"
    model_name: str = "resnet18"
    normalization: str = "mean_stdev"
    image_size: int | None = 224
    interpolation: str = "bilinear"
    seed: int = 0
    device: str = "cuda:0"
    partition: str = "default"


@dataclass
class ExperimentResult:
    """Result from a single experiment."""

    dataset: str
    model: str
    method: str
    normalization: str
    image_size: int | None
    interpolation: str
    metric_name: str
    metric_value: float
    feature_dim: int
    n_train: int
    n_val: int
    n_test: int
    seed: int
    best_lr: float
    best_use_scheduler: bool
    best_val_acc: float
    epochs_trained: int

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "model": self.model,
            "method": self.method,
            "normalization": self.normalization,
            "image_size": self.image_size,
            "interpolation": self.interpolation,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "feature_dim": self.feature_dim,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "seed": self.seed,
            "best_lr": self.best_lr,
            "best_use_scheduler": self.best_use_scheduler,
            "best_val_acc": self.best_val_acc,
            "epochs_trained": self.epochs_trained,
        }


def get_model(model_name: str, num_channels: int) -> BenchModel:
    """Instantiate a model by name."""
    if model_name == "resnet18":
        from src.models.timm import TimmPatchBenchModel

        return TimmPatchBenchModel(
            model_name="resnet18",
            num_channels=num_channels,
            pretrained=True,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def embed_dataset(
    model: BenchModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract embeddings from a dataset."""
    model.eval()
    all_features = []
    all_labels = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            features = model.forward_patch_features(images, bboxes=None)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


class LinearProbe(nn.Module):
    """Simple linear classifier."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def train_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    config: OptimizerConfig,
    device: torch.device,
    seed: int,
) -> tuple[LinearProbe, int, float, float]:
    """
    Train a linear probe using AdamW or SGD optimizer.

    Returns:
        model: Trained LinearProbe (restored to best val checkpoint)
        epochs_trained: Number of epochs actually trained
        final_train_loss: Loss on last epoch
        best_val_acc: Best validation accuracy
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = x_train.shape[1]
    model = LinearProbe(input_dim, num_classes).to(device)

    # Convert to tensors
    x_train_t = torch.from_numpy(x_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).long().to(device)
    x_val_t = torch.from_numpy(x_val).float().to(device)
    y_val_t = torch.from_numpy(y_val).long().to(device)

    # Create data loader
    train_dataset = TensorDataset(x_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )

    # Create optimizer based on config
    if config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    # Learning rate scheduler - cosine annealing
    scheduler = None
    if config.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    final_train_loss = 0.0
    epoch = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)

        final_train_loss = epoch_loss / len(x_train_t)

        # Step scheduler after each epoch
        if scheduler is not None:
            scheduler.step()

        # Validation
        model.eval()
        with torch.inference_mode():
            val_logits = model(x_val_t)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    epochs_trained = epoch + 1
    return model, epochs_trained, float(final_train_loss), float(best_val_acc)


def evaluate_probe(
    model: LinearProbe,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> float:
    """Evaluate the probe on test set."""
    model.eval()
    x_test_t = torch.from_numpy(x_test).float().to(device)

    with torch.inference_mode():
        logits = model(x_test_t)
        preds = logits.argmax(dim=1).cpu().numpy()

    return float(accuracy_score(y_test, preds))


def sweep_lr_and_scheduler(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    optimizer_name: str,
    learning_rates: list[float],
    scheduler_options: list[bool],
    device: torch.device,
    seed: int,
    epochs: int = 200,
    patience: int = 50,
    verbose: bool = True,
) -> tuple[float, float, bool, float, int]:
    """
    Sweep over learning rates and scheduler settings using validation set,
    then return the best model's test accuracy.

    Similar to how torchgeo_bench.py sweeps over C values for logistic regression.

    Returns:
        test_acc: Test accuracy of best model
        best_lr: Learning rate that achieved best validation accuracy
        best_use_scheduler: Whether scheduler was used for best model
        best_val_acc: Best validation accuracy achieved
        epochs_trained: Number of epochs trained for best model
    """
    best_lr: float | None = None
    best_use_scheduler: bool | None = None
    best_val_acc = -1.0
    best_model: LinearProbe | None = None
    best_epochs_trained = 0

    # Generate all configurations to sweep
    configs = list(product(learning_rates, scheduler_options))

    if verbose:
        logger.info(
            f"[{optimizer_name.upper()}] LR sweep over {len(configs)} configs "
            f"(train={len(x_train)}, val={len(x_val)})"
        )
        config_iterator = tqdm(configs, desc=f"{optimizer_name.upper()} configs", leave=False)
    else:
        config_iterator = configs

    for idx, (lr, use_sched) in enumerate(config_iterator):
        opt_config = OptimizerConfig(
            optimizer=optimizer_name,
            lr=lr,
            epochs=epochs,
            patience=patience,
            use_scheduler=use_sched,
        )

        model, epochs_trained, _, val_acc = train_probe(
            x_train, y_train, x_val, y_val, num_classes, opt_config, device, seed
        )

        sched_str = "sched" if use_sched else "no-sched"
        if verbose and (idx < 5 or idx % 10 == 0 or val_acc > best_val_acc):
            logger.info(f"[{optimizer_name.upper()}] lr={lr:.0e} {sched_str} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_lr = lr
            best_use_scheduler = use_sched
            best_model = model
            best_epochs_trained = epochs_trained

    assert best_lr is not None and best_model is not None, "LR sweep failed"

    sched_str = "with scheduler" if best_use_scheduler else "no scheduler"
    if verbose:
        logger.info(
            f"[{optimizer_name.upper()}] Best: lr={best_lr:.0e} ({sched_str}) val_acc={best_val_acc:.4f}"
        )

    # Evaluate best model on test set
    test_acc = evaluate_probe(best_model, x_test, y_test, device)

    if verbose:
        logger.info(f"[{optimizer_name.upper()}] Test accuracy={test_acc:.4f}")

    return test_acc, best_lr, best_use_scheduler, best_val_acc, best_epochs_trained


def run_experiment(
    exp_config: ExperimentConfig,
    optimizer_name: str,
    learning_rates: list[float],
    scheduler_options: list[bool],
    epochs: int,
    patience: int,
) -> ExperimentResult | None:
    """Run a single experiment with LR sweep."""
    device = torch.device(exp_config.device)

    # Load dataset
    result = get_datasets(
        dataset_name=exp_config.dataset_name,
        partition_name=exp_config.partition,
        batch_size=64,
        normalization=exp_config.normalization,
        return_val=True,
        image_size=exp_config.image_size,
        interpolation=exp_config.interpolation,
    )

    if result is None or len(result) != 4:
        logger.warning(f"Failed to load dataset: {exp_config.dataset_name}")
        return None

    train_dataset, train_loader, val_loader, test_loader = result

    # Get number of channels and classes
    first_sample = train_dataset[0]
    num_channels = first_sample["image"].shape[0]

    from src.datasets import NUM_CLASSES_PER_DATASET

    num_classes = NUM_CLASSES_PER_DATASET.get(exp_config.dataset_name, 10)

    # Create model
    model = get_model(exp_config.model_name, num_channels)
    model.to(device).eval()

    # Extract embeddings
    logger.info("Extracting embeddings...")
    x_train, y_train = embed_dataset(model, train_loader, device)
    x_val, y_val = embed_dataset(model, val_loader, device)
    x_test, y_test = embed_dataset(model, test_loader, device)

    feature_dim = x_train.shape[1]
    logger.info(f"Feature dim: {feature_dim}, Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    # Sweep over learning rates and scheduler options
    test_acc, best_lr, best_use_scheduler, best_val_acc, epochs_trained = sweep_lr_and_scheduler(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        num_classes,
        optimizer_name,
        learning_rates,
        scheduler_options,
        device,
        exp_config.seed,
        epochs=epochs,
        patience=patience,
        verbose=True,
    )

    method_name = f"linear-{optimizer_name}"

    return ExperimentResult(
        dataset=exp_config.dataset_name,
        model=exp_config.model_name,
        method=method_name,
        normalization=exp_config.normalization,
        image_size=exp_config.image_size,
        interpolation=exp_config.interpolation,
        metric_name="accuracy",
        metric_value=test_acc,
        feature_dim=feature_dim,
        n_train=len(x_train),
        n_val=len(x_val),
        n_test=len(x_test),
        seed=exp_config.seed,
        best_lr=best_lr,
        best_use_scheduler=best_use_scheduler,
        best_val_acc=best_val_acc,
        epochs_trained=epochs_trained,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run linear probe experiments on m-eurosat with LR sweep"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/eurosat_example/eurosat_optim_sweep_v3.csv",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200, help="Max epochs per config")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Skip completed experiments")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define sweep parameters
    normalizations = ["mean_stdev"]
    image_sizes: list[int | None] = [None, 224, 256, 448, 512]
    interpolations = ["bilinear", "bicubic", "nearest"]

    # Optimizers to test
    optimizers = ["adamw", "sgd"]

    # Learning rate sweep range
    learning_rates = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

    # Scheduler options
    scheduler_options = [True, False]

    logger.info(f"LR sweep range: {learning_rates}")
    logger.info(f"Scheduler options: {scheduler_options}")
    logger.info(f"Epochs: {args.epochs}, Patience: {args.patience}")

    # Load existing results for resume
    completed_keys: set[tuple] = set()
    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        for _, row in existing_df.iterrows():
            img_size = row.get("image_size")
            if pd.isna(img_size):
                img_size_str = "None"
            else:
                img_size_str = str(int(img_size)) if isinstance(img_size, float) else str(img_size)
            key = (
                str(row.get("normalization")),
                img_size_str,
                str(row.get("interpolation")),
                str(row.get("method")),
            )
            completed_keys.add(key)
        logger.info(f"Resume mode: Found {len(completed_keys)} completed experiments")

    # Generate experiment configurations
    experiments = []
    for norm, size, interp in product(normalizations, image_sizes, interpolations):
        # Skip interpolation variations when no resizing
        if size is None and interp != "bilinear":
            continue
        for opt in optimizers:
            experiments.append((norm, size, interp, opt))

    logger.info(f"Total experiments to run: {len(experiments)}")

    for norm, size, interp, opt in tqdm(experiments, desc="Experiments"):
        # Check resume
        size_str = "None" if size is None else str(size)
        method_name = f"linear-{opt}"
        key = (norm, size_str, interp, method_name)
        if key in completed_keys:
            logger.info(f"Skipping (already done): norm={norm}, size={size}, interp={interp}, opt={opt}")
            continue

        logger.info(f"Running: norm={norm}, size={size}, interp={interp}, opt={opt}")

        exp_config = ExperimentConfig(
            dataset_name="m-eurosat",
            model_name="resnet18",
            normalization=norm,
            image_size=size,
            interpolation=interp,
            seed=args.seed,
            device=args.device,
        )

        result = run_experiment(
            exp_config,
            optimizer_name=opt,
            learning_rates=learning_rates,
            scheduler_options=scheduler_options,
            epochs=args.epochs,
            patience=args.patience,
        )

        if result is not None:
            # Append to CSV immediately
            df = pd.DataFrame([result.to_dict()])
            df.to_csv(
                output_path,
                mode="a",
                header=not output_path.exists(),
                index=False,
            )

    logger.info(f"Experiments complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
