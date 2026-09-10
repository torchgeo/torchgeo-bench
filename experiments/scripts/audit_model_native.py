"""Report which model configs support ``dataset.normalization=model_native``.

``model_native`` is only meaningful when a model can state what its pretraining
pipeline was: pretrain statistics, a weights-bound ``Normalize``, or its own
normaliser.  Without one of those it used to fall through to a bare unit
conversion, handing the backbone raw sensor values.

Usage:
    python experiments/scripts/audit_model_native.py --out model_native_audit.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class

logger = logging.getLogger(__name__)

CONF = Path(__file__).resolve().parents[2] / "src" / "torchgeo_bench" / "conf" / "model"
SKIP_TARGETS = {"SAM3Encoder"}


def band_specs(dataset: str, bands: str):
    """Return the BandSpec list a model would receive for this dataset."""
    cls = get_bench_dataset_class(dataset)
    if bands == "rgb":
        names = set(cls.rgb_bands)
        return [b for b in cls.bands if b.name in names]
    return list(cls.bands)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="m-eurosat")
    ap.add_argument("--bands", default="rgb")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    sample = torch.rand(2, 0, 8, 8)
    for path in sorted(CONF.rglob("*.yaml")):
        config_name = str(path.relative_to(CONF).with_suffix(""))
        cfg = compose_config(model=config_name).model
        name, target = cfg.get("name"), cfg.get("_target_", "")
        if not name or not target or "coordbench" in target:
            continue
        if target.rsplit(".", 1)[-1] in SKIP_TARGETS:
            continue
        bands = band_specs(args.dataset, args.bands)
        entry: dict = {"config": config_name}
        try:
            model = instantiate(cfg, bands=bands, normalization="model_native")
            sample = torch.rand(2, len(bands), 32, 32) * 3000
            model.normalize_inputs(sample)
            entry["model_native"] = "supported"
        except ValueError as error:
            message = str(error)
            if "model_native" not in message:
                raise
            entry["model_native"] = "unsupported"
            entry["reason"] = message[:160]
        results[name] = entry
        print(f"{name:38} {entry['model_native']}", flush=True)

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    unsupported = sorted(n for n, v in results.items() if v["model_native"] == "unsupported")
    print(f"\n{len(unsupported)}/{len(results)} models do not support model_native")
    print(json.dumps(unsupported, indent=1))


if __name__ == "__main__":
    main()
