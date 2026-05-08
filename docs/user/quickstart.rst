Quickstart
==========

This page walks through running your first benchmark end-to-end:
download data, run the eval pipeline, and inspect the results.

Prerequisites
-------------

* ``torchgeo-bench`` installed (see :doc:`installation`).
* At least one GeoBench dataset downloaded into ``./data/`` — easiest is
  the small ``m-eurosat`` split.

Download data
-------------

GeoBench V1 (default) lives under ``data/geobench-1.0/``:

.. code-block:: console

   $ torchgeo-bench download
   $ # or just one V1 family by editing the script — V1 currently downloads as a bundle.

GeoBench V2 datasets can be selected individually:

.. code-block:: console

   $ torchgeo-bench download --version v2 --datasets m_eurosat

See :doc:`datasets` for the full list of supported names and the
``GEOBENCH_ROOT`` / ``GEOBENCH_V2_ROOT`` environment variables that
override the default location.

Run a benchmark
---------------

Run the default model (Random Convolutional Features) on EuroSAT V1 with
KNN-5 + linear probing + 200 bootstrap resamples:

.. code-block:: console

   $ torchgeo-bench run dataset.names=[m-eurosat]

Use a different backbone preset (anything in :file:`src/torchgeo_bench/conf/model/`):

.. code-block:: console

   $ torchgeo-bench run model=timm/resnet50 dataset.names=[m-eurosat,m-pv4ger]

Skip the (slow) linear probe and reduce bootstrap noise to iterate quickly:

.. code-block:: console

   $ torchgeo-bench run eval.skip_linear=true eval.bootstrap=100

Resume mode
-----------

If a previous run was interrupted, ``resume=true`` skips any
``(dataset, method, model, config)`` combination that already exists in the
output CSV:

.. code-block:: console

   $ torchgeo-bench run resume=true

See :doc:`results-format` for the exact key schema used by resume mode.

Inspect the results
-------------------

By default results land in ``results/all_results.csv``.  Each row is a
flat :class:`~torchgeo_bench.main.EvaluationResult`, so you can read it
directly with pandas:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("results/all_results.csv")
   print(df.groupby(["dataset", "method"])["metric_value"].mean())
