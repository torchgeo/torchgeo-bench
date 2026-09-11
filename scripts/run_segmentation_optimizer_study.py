#!/usr/bin/env python3
r"""Compare segmentation decoder optimizers using all visible GPUs and full splits.

Examples (run from the repository root in the torchgeo-bench environment)::

    python scripts/run_segmentation_optimizer_study.py --dry-run
    python scripts/run_segmentation_optimizer_study.py --gpus all --resume
    python scripts/run_segmentation_optimizer_study.py --heads linear fpn \
        --adam-lrs 0.0001 0.001 0.01 --seeds 0 1 --output-dir results/optimizer-small

Defaults: Burn Scars, pretrained ViT-S/16, RGB 224, model normalization, all five
heads, paired seeds 0/1/2. Four connections are deepest-first; patch_linear
explicitly uses just the deepest connection. The full Adam LR sweep finishes
before any L-BFGS worker launches. Adam uses constant LR and zero weight decay;
L-BFGS uses full-data, deterministic microbatch closures, strong Wolfe, persistent
history 10 and chunks of up to 20 iterations. No time budget or default cap.

Both optimizers use the same fixed, ordered microbatch membership for BN.
Adam shuffles only batch order; summed batch CE is scaled by number of batches /
total valid training pixels, including the partial last batch. Compare at the
same batch size because BN changes the objective.

Stopping monitors accepted train-mode full-training CE. After --min-iterations,
the last --patience checks (at least two) form a recent window. Historical-best
patience still accumulates small decreases, but converged/train_loss_plateau also
requires the entire window range <= max(absolute_tol, relative_tol * window_min).
A stale best with a nonflat, nonrecovering window is not_converged /
no_best_improvement, including oscillation floors. A first-half mean minus
second-half mean > tolerance means recovery: keep going even above an early best.
Equal halves exclude an odd middle observation. Plateau is not stationarity.
L-BFGS also reports gradient tolerance or numerical_stall. --max-iterations is
an optional not_converged safety cap, independent of recovery.

Features are cached once in FP32 through BenchModel; source/configuration/content
mismatches are refused. DPT requires optional transformers; real runs preflight
its decoder API before extraction, while --dry-run does not import it.

OUTPUT contains combined.csv (every requested trial, including incomplete and
failed rows), validation_selected.csv (one LR selected by mean validation across
paired seeds per head/group/optimizer), immutable trials/*/{result.json,best.pt,
final.pt,checkpoint.pt,curve.csv,progress.json,test_confusions.pt}, and state/logs.
Per-trial test mIoU image-bootstrap CIs do not measure seed/spatial uncertainty.
Never choose rates from test scores. --resume repairs interrupted block artifacts;
changing scientific settings requires a new output directory. GPU scheduling is
not scientific identity; resumed hardware changes are recorded and mixed timing
segments flagged. Keep other GPU workloads off the selected devices for timing.
"""

from _segmentation_study import main

if __name__ == "__main__":
    main("optimizer", __doc__)
