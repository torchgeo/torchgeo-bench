Glossary
========

.. glossary::
   :sorted:

   BenchDataset
       Abstract base class implemented by every dataset wrapper in
       :mod:`torchgeo_bench.datasets`.  Declares static metadata (bands,
       number of classes, task type, default split sizes) and exposes a
       :meth:`~torchgeo_bench.datasets.BenchDataset.get_dataset` factory
       returning a PyTorch ``Dataset`` for a given split.

   BenchModel
       Abstract base class implemented by every backbone in
       :mod:`torchgeo_bench.models`.  Sealed ``forward_patch_features``
       method guarantees that input normalisation is always applied
       before the subclass-specific ``_forward_patch_features``.

   bootstrap
       Resampling technique used to estimate a confidence interval
       around a metric.  ``eval.bootstrap`` controls the number of
       resamples; per-dataset CIs are reported as ``(ci_lower, ci_upper)``
       in the results CSV.

   GeoBench V1
       The original GeoBench benchmark released as a set of HDF5 files.
       Datasets in V1 use the ``m-`` prefix in the CLI
       (e.g. ``m-eurosat``) and live under ``data/classification_v1.0/``.

   GeoBench V2
       The second-generation GeoBench distribution, packaged as one
       directory per dataset under ``data/geobenchv2/``.  V2 datasets
       use no prefix (``benv2``, ``treesatai``, ``pastis``, ...).

   Hydra
       The configuration framework used to compose model and run
       configs.  See :doc:`configuration` for the override syntax and
       https://hydra.cc for the full documentation.

   intrinsic dimension
       The geometric / statistical dimension of a manifold of feature
       embeddings, estimated by methods such as TwoNN, MLE, or lPCA.
       Optional in ``torchgeo-bench`` via ``eval.intrinsic_dim`` and the
       ``[id]`` extra; see :mod:`torchgeo_bench.intrinsic_dim`.

   KNN-5
       5-nearest-neighbour classifier evaluated on backbone-extracted
       features. Reported as ``method="knn5"`` in the results CSV.

   linear probe
       Logistic regression trained on frozen backbone features. We sweep
       ``C`` over ``eval.c_range`` and report the best test-set
       performance with ``best_c``. Method label: ``linear``.

   mIoU
       Mean Intersection-over-Union, the primary metric for segmentation
       datasets.  Computed by
       :func:`~torchgeo_bench.main.evaluate_segmentation`.

   resume mode
       When ``resume=true``, the runner skips any
       ``(dataset, method, model, config)`` combination already present
       in the output CSV.  See :doc:`results-format` for the exact key.
