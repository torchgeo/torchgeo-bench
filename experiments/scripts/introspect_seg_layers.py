"""Choose segmentation-probe layers from their measured output sizes.

Layer names alone can select several outputs of the same size. Run the model and measure each candidate's output, preferring four different grid sizes and falling back to layers at different depths.

Usage:
    python experiments/scripts/introspect_seg_layers.py --out /tmp/seg_layers.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
import yaml

from torchgeo_bench.config import CONF_DIR, compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.segmentation_probe import SegmentationProbe

logger = logging.getLogger(__name__)

CONF = CONF_DIR / "model"

CANDIDATE = re.compile(r"^(.*\b(?:blocks|encoder|layers|stages|features))\.(\d+)$|^(layer)(\d+)$")

# Skip baselines without probe layers and SAM3, which needs a Hugging Face download at construction.
SKIP_TARGETS = {"ImageStatsBench", "RCFBench", "SAM3Encoder"}


def band_specs(dataset: str, bands: str):
    """Return the BandSpec list a model would receive for this dataset."""
    bench = get_bench_dataset_class(dataset)()
    return bench.select_band_specs(tuple(bench.rgb_bands) if bands == "rgb" else None)


class _Stub:
    """Provide the backbone needed by the probe's feature reshaping."""

    def __init__(self, backbone):
        self.backbone = backbone


def feature_hw(feat, backbone) -> tuple[int, int] | None:
    """Measure height and width using the probe's own reshaping rules.

    Return ``None`` for non-tensors or unsupported shapes.
    """
    if not isinstance(feat, torch.Tensor):
        return None
    try:
        processed = SegmentationProbe._process_feature(_Stub(backbone), feat)
    except ValueError:
        return None
    return int(processed.shape[-2]), int(processed.shape[-1])


def _order_key(name: str) -> tuple:
    """Sort modules by numeric path so 'blocks.9' precedes 'blocks.10'."""
    return tuple((1, int(p)) if p.isdigit() else (0, p) for p in name.split("."))


def measure(model, size: int = 224) -> dict[str, tuple[int, int]]:
    """Run one forward pass and record each candidate module's output grid."""
    seen: dict[str, tuple[int, int]] = {}
    hooks = []

    def make_hook(name):  # noqa: D401
        def hook(_module, _inp, out):
            if isinstance(out, (tuple, list)) and out:
                out = out[0]
            hw = feature_hw(out, model)
            if hw:
                seen[name] = hw

        return hook

    for name, module in model.named_modules():
        clean = name.replace("backbone.", "", 1) if name.startswith("backbone.") else name
        if CANDIDATE.match(clean):
            hooks.append(module.register_forward_hook(make_hook(clean)))
    channels = len(getattr(model, "bands", []) or []) or 3
    with torch.no_grad():
        model(torch.zeros(1, channels, size, size))
    for handle in hooks:
        handle.remove()
    return seen


def choose(seen: dict[str, tuple[int, int]]) -> tuple[list[str], str]:
    """Choose four layers, preferring different output grid sizes."""
    by_res: dict[tuple[int, int], list[str]] = {}
    for name, hw in seen.items():
        by_res.setdefault(hw, []).append(name)
    if len(by_res) >= 4:
        # Use the last layer at each size, starting with the smallest grid.
        resolutions = sorted(by_res, key=lambda hw: hw[0])[:4]
        return [sorted(by_res[r], key=_order_key)[-1] for r in resolutions], "multi-resolution"
    names = sorted(seen, key=_order_key)
    if len(names) < 4:
        return names[::-1], "few-layers"
    depth = len(names)
    idx = [depth - 1, int(depth * 0.75) - 1, int(depth * 0.5) - 1, int(depth * 0.25) - 1]
    return [names[i] for i in idx], "isotropic-evenly-spaced"


def main() -> None:
    logging.basicConfig(level=logging.ERROR)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="m-eurosat")
    ap.add_argument("--bands", default="rgb")
    ap.add_argument("--only", default=None, help="comma-separated model names")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    results: dict[str, dict] = {}
    for path in sorted(CONF.rglob("*.yaml")):
        conf = yaml.safe_load(path.read_text()) or {}
        name, target = conf.get("name"), conf.get("_target_", "")
        if not name or not target or "coordbench" in target:
            continue
        if target.rsplit(".", 1)[-1] in SKIP_TARGETS:
            continue
        if only and name not in only:
            continue
        config_name = path.relative_to(CONF).with_suffix("").as_posix()
        cfg = compose_config([f"model={config_name}"]).model
        model = instantiate(cfg, bands=band_specs(args.dataset, args.bands))
        model.eval()
        seen = measure(model)
        if not seen:
            # Record the incompatibility and fail after writing the report.
            results[name] = {
                "config": config_name,
                "unusable": "no tap produced a feature map the probe can reshape",
            }
            print(f"{name:38} UNUSABLE (probe cannot reshape its features)", flush=True)
            del model
            continue
        picks, strategy = choose(seen)
        results[name] = {
            "config": config_name,
            "existing": ((conf.get("eval") or {}).get("segmentation") or {}).get("layers"),
            "strategy": strategy,
            "layers": picks,
            "shapes": {p: list(seen[p]) for p in picks},
        }
        print(f"{name:38} {strategy:24} {picks}", flush=True)
        del model

    args.out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    unusable = sorted(n for n, v in results.items() if v.get("unusable"))
    print(f"\n{len(results)} models introspected -> {args.out}")
    if unusable:
        print(f"{len(unusable)} model(s) cannot be probed as configured: {unusable}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
