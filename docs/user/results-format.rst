Results format
==============

All evaluation runs append rows to a single CSV file (default
``results/all_results.csv``).  Each row is a flattened
:class:`~torchgeo_bench.main.EvaluationResult` describing a single
``(dataset, method, model, config)`` measurement.

Sample rows
-----------

.. code-block:: text

   dataset,method,metric_name,metric_value,ci_lower,ci_upper,feature_dim,best_c,n_train,n_val,n_test,seed,model,name,normalization,image_size,interpolation,partition,bands
   m-eurosat,knn5,accuracy,0.8234,0.8123,0.8345,512,,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,bandspec_zscore,224,bilinear,default,rgb
   m-eurosat,linear,accuracy,0.8567,0.8461,0.8673,512,0.1,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,bandspec_zscore,224,bilinear,default,rgb
   burn_scars,seg-fpn,mIoU,0.6234,0.0,0.0,768,,1000,200,300,0,torchgeo_bench.models.TimmPatchBenchModel,resnet50,bandspec_zscore,224,bilinear,default,rgb

Datasets emit unnormalized tensors; each model wrapper normalises inside
:meth:`~torchgeo_bench.models.BenchModel.normalize_inputs` according to
the strategy selected by ``cfg.dataset.normalization``.  Allowed values:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Behaviour
   * - ``bandspec_zscore``
     - Per-channel z-score using ``BandSpec`` mean/std (default).
   * - ``model_native``
     - Convert to the wrapper's ``expected_input_unit``, then apply any
       ``pretrain_mean`` / ``pretrain_std`` declared on the class.
   * - ``minmax``
     - Scale each channel to ``[0, 1]`` from BandSpec min/max.
   * - ``minmax_zscore``
     - ``minmax`` then z-score against assumed ``mean=0.5, std=0.25``.
   * - ``identity``
     - No rescaling (for models whose forward owns normalisation).

Older snapshots may carry legacy values such as ``raw`` / ``mean_stdev`` /
``percentile_2_98`` — they are kept verbatim for resume safety.

Method values
-------------

================== ==================================================================================
``method``         Meaning
================== ==================================================================================
``knn5``           KNN-5 classification (multilabel KNN for ``m-bigearthnet``).
``linear``         L-BFGS logistic regression with C-sweep on the validation set.
``intrinsic_dim``  Optional intrinsic-dimension metrics on extracted embeddings (requires
                   the ``[id]`` extra and ``eval.intrinsic_dim.enabled=true``).
``seg-<head>``     Segmentation probe with the configured head (``linear`` / ``conv_block`` /
                   ``fpn`` / ``dpt``).
================== ==================================================================================

CSV schema
----------

==================== ============================================================
Column               Description
==================== ============================================================
``dataset``          Dataset CLI name (e.g. ``m-eurosat``).
``method``           ``knn5``, ``linear``, ``intrinsic_dim``, or ``seg-<head_type>``.
``metric_name``      Primary metric (``accuracy``, ``micro_mAP``, ``mIoU``,
                     or ``id_<estimator>_<split>`` for intrinsic dim rows).
``metric_value``     Point estimate.
``ci_lower``         Bootstrap CI lower bound (0.0 when not applicable).
``ci_upper``         Bootstrap CI upper bound (0.0 when not applicable).
``feature_dim``      Embedding dimension produced by the backbone.
``best_c``           Best ``C`` from the logistic-regression sweep
                     (linear probe only, otherwise ``None``).
``best_lr``          Best learning rate (segmentation only).
``best_batch_size``  Best batch size (segmentation only).
``n_train``          Train-split sample count.
``n_val``            Validation-split sample count.
``n_test``           Test-split sample count.
``seed``             RNG seed used for the run.
``model``            Fully-qualified model class (``cfg.model._target_``).
``name``             Human-readable model name (``cfg.model.name``).
``normalization``    Strategy applied by the model wrapper (see table above).
``image_size``       Input resize size (``None`` if no resizing).
``interpolation``    Resize interpolation mode.
``partition``        GeoBench V1 partition name (``default`` for V2).
``bands``            ``rgb`` / ``all`` / a sorted comma-joined list.
``c_range_start``    ``eval.c_range[0]``.
``c_range_stop``     ``eval.c_range[1]``.
``c_range_num``      ``eval.c_range[2]``.
``merge_val``        Whether ``train+val`` was merged before final logistic fit.
``bootstrap``        Number of bootstrap resamples used for CIs.
``fw_iou``           Frequency-weighted IoU (segmentation only).
``precision``        Macro precision (segmentation only).
``recall``           Macro recall (segmentation only).
``f1``               Macro F1 (segmentation only).
==================== ============================================================

Atomic appends
--------------

Rows are appended via :func:`~torchgeo_bench.main.append_rows_atomic`,
which uses ``fcntl`` advisory file locking (available on Linux and
macOS).  This makes it safe to point multiple parallel jobs (e.g. one
per GPU or per dataset) at the same output file without corrupting it.

Resume mode
-----------

When ``resume=true``, the runner reads the existing CSV at startup and
skips any combination that already has a matching row.  The de-dup key
is:

.. code-block:: python

   (dataset, method, model._target_, model.name,
    normalization, image_size, interpolation, partition, bands)

Note that ``method`` is per-method (``knn5`` / ``linear`` /
``intrinsic_dim`` / ``seg-<head_type>``), so re-running with
``eval.skip_linear=false`` after a ``skip_linear=true`` run will fill in
just the linear-probe rows.
