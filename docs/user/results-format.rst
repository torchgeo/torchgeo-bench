Results format
==============

All evaluation runs append rows to a single CSV file (default
``results/all_results.csv``).  Each row is a flattened
:class:`~torchgeo_bench.main.EvaluationResult` describing a single
``(dataset, method, model, config)`` measurement.

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
``normalization``    Always ``raw`` after the model-normalization refactor;
                     kept for back-compat with older CSVs.
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
which uses ``fcntl`` advisory file locking.  This makes it safe to point
multiple parallel jobs (e.g. one per GPU or per dataset) at the same
output file without corrupting it.

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
