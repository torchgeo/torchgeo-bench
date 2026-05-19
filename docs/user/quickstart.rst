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

Datasets always live under ``./data/`` relative to the current working
directory (paths are fixed — there are no ``GEOBENCH_ROOT`` environment
variables).  The bundled downloader fetches each family by name:

.. code-block:: console

   $ torchgeo-bench download geobench_v1                       # all V1 classification datasets
   $ torchgeo-bench download geobench_v2                       # default V2 set (cls + seg)
   $ torchgeo-bench download geobench_v2 --datasets benv2,burn_scars
   $ torchgeo-bench download eurosat                           # torchgeo's EuroSAT mirror

See :doc:`datasets` for the full list of supported names and the
canonical destination subdirectories.

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
