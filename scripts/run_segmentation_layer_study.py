#!/usr/bin/env python3
r"""Measure intermediate-layer sensitivity with converged constant-LR Adam probes.

Examples (run from the repository root in the torchgeo-bench environment)::

    python scripts/run_segmentation_layer_study.py --dry-run
    python scripts/run_segmentation_layer_study.py --gpus all --resume
    python scripts/run_segmentation_layer_study.py --heads linear fpn \
        --layer-groups late_four spread_four early_four --adam-lrs 0.001 0.01
    python scripts/run_segmentation_layer_study.py --heads dpt \
        --layer-groups late_four spread_four early_four --output-dir results/layers-dpt
    python scripts/run_segmentation_layer_study.py --heads patch_linear \
        --layer-groups deep_single --output-dir results/layers-patch
    python scripts/run_segmentation_layer_study.py --heads linear \
        --layer-groups-json '{"deep":["blocks.11"],"middle":["blocks.5"]}'

Defaults: Burn Scars, ViT-S/16, RGB 224, linear/conv_block/fpn, paired seeds
0/1/2 and rates 1e-4/1e-3/1e-2. Presets are ordered deepest-first:
deep_single=[11], late_single=[8], mid_single=[5], shallow_single=[2],
late_four=[11,10,9,8], spread_four=[11,8,5,2], early_four=[5,4,3,2].
Numbers denote blocks.N. All required layers share one union feature cache:
the backbone sees each sample once; workers select ordered tensor subsets.
Custom models need explicit compatible groups. DPT requires four layers and
patch_linear exactly one; incompatible grids fail, never silently substitute.
DPT's optional transformers dependency is preflighted before real extraction;
--dry-run does not import it.

Four-versus-four is the primary connection comparison: one-versus-four also
changes decoder capacity. Both layer count and actual parameter count are saved.
The optimizer study's accepted-training-CE convergence, deterministic FP32,
train-only BN calibration, checkpoint resumption and timing protocol apply.
Adam shuffles only the order of fixed BN microbatches, with summed CE scaled by
number of batches / total valid pixels (also for a partial last batch).
After --min-iterations, a recent window of --patience checks (at least two) must
have range <= max(absolute_tol, relative_tol * window_min) for convergence, as
well as stale cumulative-best patience. A nonflat window with stale best and no
material first-half-to-second-half mean decrease is not_converged /
no_best_improvement. Equal halves exclude an odd middle observation. Materially
recovering windows continue even above the historical best; a flat window is
operational convergence, not stationarity or an oscillation-floor certificate.
No time budget/default cap; optional --max-iterations is labeled not_converged.

OUTPUT/combined.csv preserves all raw trials and per-image test-bootstrap CIs.
OUTPUT/validation_selected.csv selects one LR by mean validation over paired
seeds for each head/group/optimizer, then reports mean test mIoU and seed SD.
Incomplete grids are labeled, not silently compared. Test never selects anything.
Checkpoint/progress/curve and per-job logs are kept under OUTPUT/trials and state.
--resume requires exact scientific identity but permits different GPU scheduling,
recording hardware changes separately. Avoid other workloads on selected GPUs.
"""

from _segmentation_study import main

if __name__ == "__main__":
    main("layer", __doc__)
