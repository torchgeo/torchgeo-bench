"""Overfitting sanity check CLI — pre-screening tool for segmentation encoders.

Run with:
    torchgeo-bench overfit-check model=resnet50 dataset.names=[m-eurosat]

Or directly:
    python -m torchgeo_bench.overfit_check model=resnet50 dataset.names=[m-eurosat]
"""

import logging
import os

import hydra
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torchgeo.datasets.errors import DatasetNotFoundError

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.main import _build_seg_probe_and_solver, _expand_dataset_list
from torchgeo_bench.sanity_checks import run_overfit_check

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def overfit_check(cfg: DictConfig) -> None:
    """Run overfitting sanity checks for segmentation encoders.

    For each (model, dataset) pair, trains a fresh probe head on 1–2 training
    batches and verifies it can achieve near-perfect mIoU on those same batches.
    Results are written to check.output as a CSV.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    check_cfg = cfg.get("check", {})
    output_path: str = check_cfg.get("output", "overfit_check_results.csv")
    device = torch.device(cfg.device)

    dataset_names = _expand_dataset_list(cfg.dataset.names)
    results: list[dict] = []
    n_pass = 0
    n_total = 0

    for ds_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(ds_name)
        except KeyError:
            logger.warning(f"[{ds_name}] Skipping — not in dataset registry.")
            continue

        if ds_cls.task != "segmentation":
            logger.info(f"[{ds_name}] Skipping — not a segmentation dataset.")
            continue

        try:
            train_dataset, train_loader, _val_loader, _test_loader = get_datasets(
                dataset_name=ds_name,
                partition_name=cfg.dataset.partition,
                batch_size=cfg.dataset.batch_size,
                return_val=True,
                image_size=getattr(cfg.dataset, "image_size", None),
                interpolation=getattr(cfg.dataset, "interpolation", "bicubic"),
                bands=getattr(cfg.dataset, "bands", "rgb"),
            )
        except (FileNotFoundError, DatasetNotFoundError) as exc:
            logger.warning(f"[{ds_name}] Skipping — dataset not found: {exc}")
            continue
        num_channels = train_dataset[0]["image"].shape[0]
        num_classes = ds_cls.num_classes
        ignore_index = 255

        # Instantiate model
        model_cfg = OmegaConf.merge(cfg.model, {"num_channels": num_channels})
        model = instantiate(model_cfg)
        model.to(device).eval()

        # Build segmentation probe (same path as main benchmark)
        eval_cfg = cfg.eval
        if "eval" in cfg.model and cfg.model.eval is not None:
            eval_cfg = OmegaConf.merge(eval_cfg, cfg.model.eval)

        probe, _solver = _build_seg_probe_and_solver(
            model=model,
            num_classes=num_classes,
            eval_cfg=eval_cfg,
            device=device,
            lr=eval_cfg.segmentation.lr,
        )

        logger.info(f"[{ds_name}] Running overfit check for model={cfg.model.name} ...")
        result = run_overfit_check(
            probe=probe,
            train_loader=train_loader,
            num_classes=num_classes,
            device=device,
            check_cfg=check_cfg,
            ignore_index=ignore_index,
        )

        row = {
            "model": cfg.model.name,
            "dataset": ds_name,
            **result,
        }
        results.append(row)
        n_total += 1
        if result["passed"]:
            n_pass += 1

        # Free GPU memory between datasets
        del model, probe, _solver
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if not results:
        logger.warning("No segmentation datasets evaluated. Check dataset.names and data paths.")
        return

    df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Overfit check complete: {n_pass}/{n_total} passed. Results written to {output_path}")


if __name__ == "__main__":
    overfit_check()
