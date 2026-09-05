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
import yaml
from hydra.errors import InstantiationException
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="m-eurosat")
    ap.add_argument("--bands", default="rgb")
    args = ap.parse_args()

    results: dict[str, dict] = {}
    sample = torch.rand(2, 0, 8, 8)
    for path in sorted(CONF.rglob("*.yaml")):
        conf = yaml.safe_load(path.read_text()) or {}
        name, target = conf.get("name"), conf.get("_target_", "")
        if not name or not target or "coordbench" in target:
            continue
        if target.rsplit(".", 1)[-1] in SKIP_TARGETS:
            continue
        bands = band_specs(args.dataset, args.bands)
        # Model configs may interpolate root keys (rcf.yaml has `seed: ${seed}`),
        # so compose under a root the way Hydra does rather than standalone.
        root = OmegaConf.create(
            {"seed": 0, "model": {k: v for k, v in conf.items() if k != "eval"}}
        )
        cfg = root.model
        entry: dict = {"config": str(path.relative_to(CONF).with_suffix(""))}
        try:
            model = instantiate(cfg, bands=bands, normalization="model_native", _convert_="object")
            sample = torch.rand(2, len(bands), 32, 32) * 3000
            model.normalize_inputs(sample)
            entry["model_native"] = "supported"
        except (ValueError, InstantiationException) as exc:
            # Hydra wraps construction errors, so unwrap before deciding.  Only
            # the "model cannot state its pretraining pipeline" case is a
            # classification; anything else is a real bug and must surface.
            cause = exc.__cause__ if isinstance(exc, InstantiationException) else exc
            message = str(cause)
            if not isinstance(cause, ValueError) or "model_native" not in message:
                raise
            entry["model_native"] = "unsupported"
            entry["reason"] = message[:160]
        results[name] = entry
        logger.info("%-38s %s", name, entry["model_native"])

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    unsupported = sorted(n for n, v in results.items() if v["model_native"] == "unsupported")
    logger.info("%d/%d models do not support model_native", len(unsupported), len(results))
    print(json.dumps(unsupported, indent=1))  # noqa: T201


if __name__ == "__main__":
    main()
