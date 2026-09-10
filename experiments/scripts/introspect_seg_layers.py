"""Derive segmentation-probe layers for each model config by measurement.

``SegmentationProbe`` taps modules by ``named_modules()`` name and builds a
multi-scale head from whatever those taps emit.  Choosing taps by name alone is
unreliable: for a hierarchical CNN the deepest repeated container is the block
list *inside one stage*, so all four taps come back at the same spatial
resolution and the "multi-scale" probe is nothing of the sort.

This hooks every candidate module, runs one real forward pass, records the
feature shape each module actually produces, and picks taps by measured
resolution — falling back to evenly spaced blocks for isotropic ViTs, which
genuinely have only one resolution.

Usage:
    python experiments/scripts/introspect_seg_layers.py --out /tmp/seg_layers.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch

from torchgeo_bench.config import instantiate
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.segmentation_probe import SegmentationProbe
from torchgeo_bench.settings import load_yaml

logger = logging.getLogger(__name__)

CONF = Path(__file__).resolve().parents[2] / "src" / "torchgeo_bench" / "conf" / "model"

CANDIDATE = re.compile(r"^(.*\b(?:blocks|encoder|layers|stages|features))\.(\d+)$|^(layer)(\d+)$")

# Not segmentation-probe candidates: location encoders take (lon, lat), the
# statistical baselines have no backbone, and SAM3 fetches its checkpoint from
# HuggingFace at construction, which the compute nodes cannot reach.
SKIP_TARGETS = {"ImageStatsBench", "RCFBench", "SAM3Encoder"}


def band_specs(dataset: str, bands: str):
    """Return the BandSpec list a model would receive for this dataset."""
    cls = get_bench_dataset_class(dataset)
    if bands == "rgb":
        names = set(cls.rgb_bands)
        return [b for b in cls.bands if b.name in names]
    return list(cls.bands)


class _Stub:
    """Minimal stand-in so SegmentationProbe._process_feature can be reused."""

    def __init__(self, backbone):
        self.backbone = backbone


def feature_hw(feat, backbone) -> tuple[int, int] | None:
    """Return the (H, W) the probe would see, using the probe's own reshape.

    Reimplementing the reshape here would measure something the probe does not
    actually do, so call its method directly.  A feature it cannot reshape
    raises there and is reported as an incompatibility rather than skipped.
    """
    if not isinstance(feat, torch.Tensor):
        return None
    try:
        processed = SegmentationProbe._process_feature(_Stub(backbone), feat)
    except ValueError:
        # The probe itself refuses this tensor; record it as unusable so the
        # model is reported rather than silently tapped somewhere else.
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
    """Pick four taps, preferring distinct spatial resolutions."""
    by_res: dict[tuple[int, int], list[str]] = {}
    for name, hw in seen.items():
        by_res.setdefault(hw, []).append(name)
    if len(by_res) >= 4:
        # Deepest module at each resolution, coarsest grid first.
        resolutions = sorted(by_res, key=lambda hw: hw[0])[:4]
        return [sorted(by_res[r], key=_order_key)[-1] for r in resolutions], "multi-resolution"
    names = sorted(seen, key=_order_key)
    if len(names) < 4:
        return names[::-1], "few-layers"
    depth = len(names)
    idx = [depth - 1, int(depth * 0.75) - 1, int(depth * 0.5) - 1, int(depth * 0.25) - 1]
    return [names[i] for i in idx], "isotropic-evenly-spaced"


def main() -> None:
    """Entry point."""
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
        conf = load_yaml(path) or {}
        name, target = conf.get("name"), conf.get("_target_", "")
        if not name or not target or "coordbench" in target:
            continue
        if target.rsplit(".", 1)[-1] in SKIP_TARGETS:
            continue
        if only and name not in only:
            continue
        cfg = {key: value for key, value in conf.items() if key != "eval"}
        model = instantiate(cfg, bands=band_specs(args.dataset, args.bands))
        model.eval()
        seen = measure(model)
        if not seen:
            # Every tap produced a tensor SegmentationProbe refuses to reshape
            # (e.g. OlmoEarth v1's 2352 = 28^2 x 3 grouped tokens).  Record it
            # as a real incompatibility and fail the run at the end.
            results[name] = {
                "config": str(path.relative_to(CONF).with_suffix("")),
                "unusable": "no tap produced a feature map the probe can reshape",
            }
            print(f"{name:38} UNUSABLE (probe cannot reshape its features)", flush=True)
            del model
            continue
        picks, strategy = choose(seen)
        results[name] = {
            "config": str(path.relative_to(CONF).with_suffix("")),
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
