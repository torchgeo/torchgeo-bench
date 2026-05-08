Evaluation methodology
======================

This page is a quick summary; see `METHODOLOGY.md
<https://github.com/torchgeo/torchgeo-bench/blob/main/METHODOLOGY.md>`__
in the repository root for the full formal description.

The benchmark always evaluates a **frozen** backbone.  We measure the
quality of learned representations, not end-to-end fine-tuning
performance.  The model is reinstantiated fresh for each dataset
because ``num_channels`` (and therefore the input convolution) varies
across datasets and band selections.

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Dataset type
     - Methods
     - Primary metric
   * - Classification
     - ``knn5``, ``linear``
     - accuracy / micro-mAP
   * - Segmentation
     - ``seg-{linear,conv_block,fpn,dpt}``
     - mIoU

Classification
--------------

For every classification dataset:

1. **Feature extraction** runs the frozen backbone over the train, val,
   and test loaders once with ``torch.inference_mode()``.  ViT-style
   3-D ``(B, tokens, K)`` outputs are mean-pooled across the token
   dimension; dictionary outputs are flattened on the standard keys
   (``"norm"``, ``"global_pool"``, ``"head.global_pool"``).
2. **KNN-5**: a fixed-k=5 FAISS ``IndexFlatL2`` is fit on the train
   embeddings and evaluated on the test set, with bootstrapped 95%
   CIs.  No hyperparameter tuning, no validation use.
3. **Linear probe**: L-BFGS logistic regression is swept over
   ``C ∈ logspace(eval.c_range)`` on the validation set.  The best
   ``C`` is then optionally re-fit on ``train ∪ val``
   (``eval.merge_val=true``) and evaluated on test, with bootstrapped
   95% CIs.
4. **Multilabel datasets** (``m-bigearthnet``) use a multilabel KNN and
   linear probe; the reported metric is ``micro_mAP``.
5. **Optional intrinsic dimension** (when ``eval.intrinsic_dim.enabled=true``)
   computes one row per ``(estimator, split)`` from the same extracted
   embeddings.  See :doc:`/api/intrinsic_dim`.

Segmentation
------------

For every segmentation dataset:

1. ``SegmentationProbe`` registers forward hooks on the configured
   :attr:`eval.segmentation.layers <torchgeo_bench.segmentation_probe.SegmentationProbe>`
   (see :doc:`segmentation-layers` for verified values per backbone
   family).
2. The chosen head — ``linear`` / ``conv_block`` / ``fpn`` / ``dpt`` —
   is trained on top of the frozen backbone with AdamW and
   ``CrossEntropyLoss(ignore_index=255)`` for ``eval.segmentation.epochs``
   epochs.
3. By default the backbone features are extracted **once** and cached in
   RAM (``eval.segmentation.cache_features=true``,
   ``cache_dtype=float16``) so the head training pass does not re-run
   the backbone.  Disable caching if RAM is the bottleneck.
4. Evaluation uses ``MulticlassJaccardIndex`` from `torchmetrics
   <https://lightning.ai/docs/torchmetrics/stable/>`__ (mIoU) plus
   frequency-weighted IoU and macro precision / recall / F1 (rolled
   into the result row, see :doc:`results-format`).

For the precise classifier construction, bootstrap procedure, and the
caching contract, see ``METHODOLOGY.md``.