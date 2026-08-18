#!/usr/bin/env python
"""Audit which sweep models support model_native normalization.

Instantiates every primary-sweep model with ``normalization=model_native`` and
pushes one tiny random batch through it on CPU; records init-time
("requires expected_input_unit") and forward-time ("declares no pretrain
stats") failures. Uses each model's m-eurosat band selection (all 47 models
cover m-eurosat).

Output TSV: model, status(ok|init_fail|forward_fail), error.
"""

import json
import pathlib
import sys
import traceback

import torch
from hydra import compose, initialize_config_module

HERE = pathlib.Path(__file__).parent


def main() -> int:
    from torchgeo_bench.config import instantiate
    from torchgeo_bench.datasets import get_bench_dataset_class
    from torchgeo_bench.main import resolve_model_config

    name2cfg = json.loads((HERE / "name2cfg.json").read_text())
    bands_map = {}
    for row in (HERE / "bands_map.tsv").read_text().splitlines():
        model, dataset, bands = row.split("\t")
        if dataset == "m-eurosat":
            bands_map[model] = bands

    ds_cls = get_bench_dataset_class("m-eurosat")
    bench = ds_cls()

    out = []
    for name, cfg_name in sorted(name2cfg.items()):
        bands = bands_map.get(name)
        if bands is None:
            out.append((name, "skip", "no m-eurosat bands"))
            continue
        overrides = [
            f"+model={cfg_name}",
            "dataset.names=[m-eurosat]",
            f"dataset.bands={bands if bands in ('rgb', 'all') else '[' + bands + ']'}",
            "dataset.normalization=model_native",
            "device=cpu",
        ]
        try:
            with initialize_config_module(config_module="torchgeo_bench.conf", version_base=None):
                cfg = compose(config_name="config", overrides=overrides)
            model_cfg = resolve_model_config(cfg.model, "m-eurosat")
            bands_resolved = (
                tuple(bench.rgb_bands)
                if bands == "rgb"
                else None
                if bands == "all"
                else tuple(bands.split(","))
            )
            bands_list = bench.select_band_specs(bands_resolved)
            model_cfg.pop("interpolation", None)
            kwargs = {"bands": bands_list, "normalization": "model_native"}
            if model_cfg.get("mode", None) == "empirical":
                out.append((name, "skip", "empirical rcf needs dataset"))
                continue
            model = instantiate(model_cfg, **kwargs)
        except Exception as exc:  # audit: classify, don't crash
            traceback.print_exc()
            out.append((name, "init_fail", f"{type(exc).__name__}: {exc}"))
            continue
        try:
            model.eval()
            size = 224
            x = torch.rand(1, len(bands_list), size, size) * 1000
            with torch.inference_mode():
                model(x)
            out.append((name, "ok", ""))
        except Exception as exc:  # audit: classify, don't crash
            traceback.print_exc()
            out.append((name, "forward_fail", f"{type(exc).__name__}: {exc}"))
        del model
        print(f"{name}: {out[-1][1]}", flush=True)

    report = HERE / "model_native_audit.tsv"
    report.write_text("\n".join("\t".join(r) for r in out) + "\n")
    print(f"wrote {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
