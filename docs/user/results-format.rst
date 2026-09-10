Results format
==============

All evaluation runs append rows to per-model CSV files.  Each row is a
flattened :class:`~torchgeo_bench.main.EvaluationResult` describing a single
``(dataset, method, model, config)`` measurement.  Rows are split by *kind*
across three directories, keyed by ``<model name>.csv``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Directory
     - Row ``method``\ s
     - Contents
   * - ``results/models/``
     - ``knn5``, ``linear``, ``seg-*``
     - Classification/segmentation metrics -- rewritten on every metrics rerun.
   * - ``results/profiles/``
     - ``profile``
     - Throughput/latency/param-count measurements -- one-time per model+hardware.
   * - ``results/intrinsic_dim/``
     - ``intrinsic_dim``
     - Intrinsic-dimension estimates -- one-time per model.

Profile and intrinsic-dim rows are one-time model+hardware measurements, so
they are kept out of the metrics file: rerunning a classification sweep
(new dataset, fixed metric, etc.) only touches ``results/models/``, instead
of also rewriting/diffing the expensive one-off profile and intrinsic-dim
rows every time.

Sample rows
-----------

.. code-block:: text

   dataset,method,metric_name,metric_value,ci_lower,ci_upper,feature_dim,best_c,n_train,n_val,n_test,seed,model,name,normalization,image_size,interpolation,partition,bands,num_classes
   m-eurosat,knn5,accuracy,0.8234,0.8123,0.8345,512,,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,bandspec_zscore,224,bilinear,default,rgb,10
   m-eurosat,linear,accuracy,0.8567,0.8461,0.8673,512,0.1,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,bandspec_zscore,224,bilinear,default,rgb,10
   burn_scars,seg-fpn,mIoU,0.6234,0.0,0.0,768,,1000,200,300,0,torchgeo_bench.models.TimmPatchBenchModel,resnet50,bandspec_zscore,224,bilinear,default,rgb,3

Datasets emit unnormalized tensors; each model wrapper normalises inside
:meth:`~torchgeo_bench.models.BenchModel.normalize_inputs` according to
the strategy selected by ``--normalization`` (``cfg.dataset.normalization``
internally).  Allowed values:

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
``knn5``           KNN-5 classification (multilabel KNN for ``m-bigearthnet``). -> ``results/models/``
``linear``         L-BFGS logistic regression with C-sweep on the validation set. -> ``results/models/``
``intrinsic_dim``  Optional intrinsic-dimension metrics on extracted embeddings (requires
                   the ``[id]`` extra) plus dependency-free centered feature-spectrum
                   diagnostics. Both are emitted when running via the
                   ``torchgeo-bench intrinsic-dim`` subcommand (internally
                   ``eval.intrinsic_dim.enabled=true``); set ``estimators=[]`` for
                   spectrum-only output without ``torchid``. -> ``results/intrinsic_dim/``
``profile``        Optional throughput/latency/param-count measurement, produced by the
                   ``torchgeo-bench profile`` subcommand (internally
                   ``eval.profile.enabled=true``). -> ``results/profiles/``
``seg-<head>``     Segmentation probe with the configured head (``linear`` / ``conv_block`` /
                   ``fpn`` / ``dpt``). -> ``results/models/``
================== ==================================================================================

CSV schema
----------

==================== ============================================================
Column               Description
==================== ============================================================
``dataset``          Dataset CLI name (e.g. ``m-eurosat``).
``method``           ``knn5``, ``linear``, ``intrinsic_dim``, or ``seg-<head_type>``.
``metric_name``      Primary metric (``accuracy``, ``micro_mAP``, ``mIoU``,
                     ``id_<estimator>_<split>`` for intrinsic-dimension rows,
                     or ``spectrum_<metric>_<split>`` for feature-spectrum rows).
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
``num_classes``      Dataset label count. It is also part of the resume key so
                     label-schema changes cannot reuse stale rows.
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

When ``--resume`` is passed, the runner reads the existing CSV(s) at startup and
skips any combination that already has a matching row.  Since profile and
intrinsic-dim rows may live in their own files (see above), resume reads
all three files -- ``results/models/<name>.csv``,
``results/profiles/<name>.csv``, and ``results/intrinsic_dim/<name>.csv``
-- and unions their completed-metric keys.  The de-dup key is:

.. code-block:: python

   (dataset, method, model._target_, model.name,
    normalization, image_size, interpolation, partition, bands, num_classes)

Note that ``method`` is per-method (``knn5`` / ``linear`` /
``intrinsic_dim`` / ``seg-<head_type>``), so re-running without
``--skip-linear`` after a ``--skip-linear`` run will fill in
just the linear-probe rows.

Rows written before version 0.5.0 do not have ``num_classes`` and are treated
as incomplete by resume mode. The checked-in SpaceNet2/7 rows produced under
the old three-class task were removed; they remain available from the 0.4.0 tag.

CoordBench uses a separate schema and output path. See :doc:`coordbench` and
:class:`~torchgeo_bench.coordbench.CoordResult`.
