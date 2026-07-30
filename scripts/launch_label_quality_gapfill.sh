#!/bin/bash
# Gap-fill for the 2026-07-28 label_quality_v3 sweep.
#
# WHY THIS EXISTS: that sweep launched 24 (model x dataset) pairs across 6 nodes
# all appending to one CSV with resume=true, but `run.py::_completed` keyed its
# resume-skip on (dataset, method) WITHOUT the model. So whichever backbone
# wrote a dataset first suppressed every other backbone's rows for it -- ~14
# pairs ran their full OOF ensemble to completion and then silently wrote
# nothing. `_completed` now keys on (dataset, model, method, bands); this script
# re-runs only what is missing from the CSV.
#
# THE RE-RUN IS CHEAP. Fold checkpoints were always model-keyed (FM-1) and were
# written even by the pairs whose rows were dropped: every (model, dataset) has
# a complete 25/25 checkpoint set except spacenet2. With resume=true each pair
# reloads its 25 folds and recomputes scores instead of retraining -- minutes
# per pair, not the ~10-12 h the original run took. spacenet2 is the exception
# and is deliberately NOT included here (see below).
#
# SCOPE -- 16 pairs. Excluded on purpose:
#   * anything already complete in the CSV (both methods present);
#   * the 2 convnet spacenet2 pairs, which are retrains, not rescores -- see the
#     tail of this file.
#
# Jobs 14151577/14151578 (an earlier revision of this header called them "still
# RUNNING") have since finished. Their outcomes differ, and so does their
# handling here:
#   * vit_base/spacenet2 completed 25/25 checkpoints and only lost its rows to
#     the model-blind guard, so it IS a cheap rescore -- included in shard 4.
#   * convnext_base/spacenet2 OOM-died at fold 19/25 and is NOT included.
#
# convnext_base/fotw is included: it has cleanlab rows but no aer (it raced
# into the gap between resnet50's two method writes), and the guard is
# per-method, so re-running appends only the missing aer half.
#
# This run also populates the new degeneracy columns (degenerate,
# min_class_coverage, oof_per_class_iou_min, score_iqr) for every cell it
# touches, so no separate rescore pass is needed for them.
#
# Sharding mirrors launch_label_quality_tuned.sh: one node per shard, 4 GPUs,
# one pair pinned per GPU. Since every pair here is checkpoint-resumed, the
# walltime is small -- 2 h is generous.
#
# Hparams still come from the model configs, NOT from here (see the tuned
# launcher's header for why passing them as env vars would be wrong).
set -euo pipefail
cd /p/project1/hai_uqmethodbox/nils/torchgeo-bench

export BANDS=rgb
export RESUME=true
export OUTPUT=results/label_quality_v3/label_quality_results.csv

R="timm/resnet50"
C="timm/convnext_base"
V="timm/vit/vit_base_patch16_224"
D="timm/vit/vit_large_patch16_dinov3sat"

# Shard 1 -- resnet50 leftovers + the convnext aer-only repair (4 pairs).
PAIRS_LIST="$R cloudsen12; $R flair2; $R spacenet7; $C fotw" \
  sbatch --time=02:00:00 --job-name=lq_gap_1 scripts/slurm_label_quality_sweep.sh

# Shard 2 -- convnext leftovers (4 pairs).
PAIRS_LIST="$C caffe; $C cloudsen12; $C flair2; $C spacenet7" \
  sbatch --time=02:00:00 --job-name=lq_gap_2 scripts/slurm_label_quality_sweep.sh

# Shard 3 -- vit_base leftovers (4 pairs).
PAIRS_LIST="$V caffe; $V cloudsen12; $V flair2; $V fotw" \
  sbatch --time=02:00:00 --job-name=lq_gap_3 scripts/slurm_label_quality_sweep.sh

# Shard 4 -- vit_base spacenet7 + spacenet2 + dinov3sat leftovers (4 pairs).
# vit_base/spacenet2 belongs here, not with the retrains: it has 25/25
# checkpoints and only lost its rows to the guard, so it rescores in minutes.
PAIRS_LIST="$V spacenet7; $V spacenet2; $D caffe; $D fotw" \
  sbatch --time=02:00:00 --job-name=lq_gap_4 scripts/slurm_label_quality_sweep.sh

# AFTERWARDS -- the 2 convnet spacenet2 cells, which this script does not cover.
# Both OOM'd at finetune_batch_size=64 (resnet50 stopped at 9/25 checkpoints,
# convnext_base at 19/25), so they are retrains, not rescores.
#
# The cause is tile size, not model size: only the ViTs set `auto_resize: true`
# and downsample spacenet2's ~650px tiles to 224, so the convnets train at the
# native frame -- ~8.4x the pixels.
#
# Their stale batch-64 checkpoints must be DELETED first, or resume=true reloads
# them and the cell silently mixes batch sizes across its folds:
#
#   rm -rf results/label_quality_v3/label_quality/spacenet2/checkpoints/resnet50 \
#          results/label_quality_v3/label_quality/spacenet2/checkpoints/convnext_base
#
#   EXTRA_ARGS="model.eval.segmentation.finetune_batch_size=32" \
#   PAIRS_LIST="timm/resnet50 spacenet2; timm/convnext_base spacenet2" \
#     sbatch --time=18:00:00 --job-name=lq_gap_sn2 scripts/slurm_label_quality_sweep.sh
#
# COMPARABILITY CAVEAT: both cells then run at batch 32 while the same models'
# other five datasets ran at 64. That (lr, batch) combination was never sampled
# by the hparam search, and spacenet2 is exactly where resnet50 was still
# improving at the end of the budget (+0.0218 mIoU from step 1500->2000).
# Footnote these two cells wherever these numbers get written up.
