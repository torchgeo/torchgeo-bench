Troubleshooting
===============

``Dataset directory not found`` / files missing
-----------------------------------------------

Datasets must live under ``./data/`` from the directory where you run
``torchgeo-bench``.  The runner does **not** honour ``GEOBENCH_ROOT``
or ``GEOBENCH_V2_ROOT`` environment variables — paths are fixed:

* V1: :file:`data/classification_v1.0/<name>/`
* V2: :file:`data/geobenchv2/<name>/`
* EuroSAT: :file:`data/eurosat/`

Re-run ``torchgeo-bench download …`` to fetch missing data.  If your
data lives elsewhere, symlink ``data/`` to the real location.

``ModuleNotFoundError: geobench``
---------------------------------

The legacy ``geobench`` package is no longer a dependency.  V1 datasets
are read directly from HDF5 (the internal ``GeoBenchv1`` loader in
:file:`src/torchgeo_bench/datasets/geobench_v1.py`); V2 dispatches to
upstream ``geobench_v2.datasets.GeoBench<X>``.  Make sure your
environment matches the pinned ``geobenchv2`` version in
:file:`pyproject.toml`.

CUDA out of memory
------------------

.. code-block:: console

   $ torchgeo-bench run dataset.batch_size=32
   $ # or run on CPU
   $ torchgeo-bench run device=cpu

For segmentation, also try

.. code-block:: console

   $ torchgeo-bench run \
       eval.segmentation.cache_dtype=float32 \
       eval.segmentation.cache_features=false

if RAM (rather than GPU memory) is the bottleneck.

``KeyError: 's2'`` on a V2 dataset
----------------------------------

A known V2 issue: ``geobench_v2.rearrange_bands`` expects modality keys
(``'s2'``, ``'s1'``, …) that aren't present when a flat band list is
requested.  Workaround: use ``dataset.bands=all`` for affected V2
datasets.

Build / docs warnings
---------------------

If you build the docs locally without internet access, expect ~9
``WARNING: failed to reach any of the inventories`` messages from
``sphinx.ext.intersphinx``.  These are network reachability errors, not
real issues — Read the Docs builds with network access and resolves
these inventories cleanly.