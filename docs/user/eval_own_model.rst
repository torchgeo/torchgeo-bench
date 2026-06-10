Evaluate your own model (Stage 1)
==================================

This guide lets you benchmark any frozen pretrained geospatial model against
the GeoBench datasets locally — with no pull request required.  Follow
:doc:`contribute_model` (Stage 2) once you are happy with the results and want
to contribute the model to the shared benchmark.

.. _eval-prerequisites:

Prerequisites
-------------

Clone the repository, activate the environment, and install the package:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ conda activate torchgeo-bench
   $ uv sync

If your model requires optional dependencies (e.g. a special
:doc:`model library </user/models>` or a custom tokenizer), install the
matching extra:

.. code-block:: console

   $ uv sync --extra myextra   # e.g. --extra olmoearth, --extra sam3

Then download one or more GeoBench datasets:

.. code-block:: console

   $ torchgeo-bench download geobench_v1          # all V1 classification (≈5 GB)
   $ torchgeo-bench download geobench_v2          # all V2 cls + seg (≈40 GB)
   $ torchgeo-bench download geobench_v2 --datasets benv2,burn_scars  # subset

See :doc:`datasets` for the full list of dataset names and canonical sensor
coverage for each.

.. _eval-implement:

Implement your model
--------------------

Copy the template file to your working directory and fill in the ``TODO``
sections:

.. code-block:: console

   $ cp src/torchgeo_bench/models/contrib_template.py ./my_geofm.py

The template ships two skeleton classes:

* ``MyGeoFM`` — standard case, uses the default ``bandspec_zscore``
  normalization.
* ``MyGeoFMInternal`` — identity-normalization variant, for backbones that
  handle preprocessing internally (e.g. they ship their own ``Normalizer``
  module or always expect raw sensor values).

**Normalization strategy decision table**

Pick the strategy that matches how your backbone was trained:

.. list-table::
   :header-rows: 1
   :widths: 22 40 38

   * - Strategy
     - When to use
     - How to set it
   * - ``bandspec_zscore``
     - Most remote-sensing backbones (pre-trained on normalized inputs).
       Produces ~N(0, 1) features regardless of source sensor unit.  Safe
       default when you are unsure.
     - Default in ``MyGeoFM``; pass
       ``normalization="bandspec_zscore"`` to ``super().__init__`` or leave
       it out entirely.
   * - ``identity``
     - Backbone ships its own normalizer (e.g. built-in ``Normalize`` layer,
       OlmoEarth-style ``Normalizer`` module) or is always fed raw DN / float
       values.  The framework must *not* apply a second normalization on top.
     - Use ``MyGeoFMInternal`` template, which passes
       ``normalization="identity"`` to ``super().__init__``.  The sealed
       ``forward_patch_features`` then calls ``_forward_patch_features``
       with the unchanged tensor.
   * - ``model_native``
     - Pre-train mean/std are known (e.g. ImageNet RGB stats, or published
       per-channel stats for your dataset).  The framework converts the raw
       sensor units to the backbone's expected unit first, then applies the
       declared mean/std.
     - Set ``expected_input_unit``, ``pretrain_mean``, and ``pretrain_std``
       as class attributes *before* calling ``super().__init__(bands=bands)``.
       See :class:`~torchgeo_bench.models.TimmPatchBenchModel` for a
       real-world example.

For the full list of available strategies and their exact semantics, see
:file:`src/torchgeo_bench/models/_normalization.py`.

.. _eval-hydra-config:

Create a Hydra config
---------------------

Drop a YAML file at :file:`src/torchgeo_bench/conf/model/my_model.yaml`.
The only required key is ``_target_``, which must point to your class:

.. code-block:: yaml

   # src/torchgeo_bench/conf/model/my_model.yaml
   _target_: my_geofm.MyGeoFM    # dotted import path to your class
   pretrained: true
   name: my_model                 # human-readable label in the results CSV

   # Add any kwargs your __init__ accepts (except `bands` — see note below).
   # embed_dim: 768
   # checkpoint: path/to/weights.pt

.. note::

   **Do not put** ``bands`` **in the YAML.**  The runner reads the current
   dataset's :class:`~torchgeo_bench.datasets.base.BandSpec` list at runtime
   and injects it into the constructor automatically.  Adding it to the YAML
   will cause a ``TypeError`` (duplicate keyword argument).

If your class is not importable from the default Python path, add the
parent directory to ``PYTHONPATH`` before running:

.. code-block:: console

   $ export PYTHONPATH="$PWD:$PYTHONPATH"

.. _eval-run:

Run the benchmark
-----------------

Pass your config name as ``model=my_model`` to the ``run`` subcommand:

.. code-block:: console

   $ torchgeo-bench run model=my_model dataset.names=[m-eurosat]

Run multiple datasets in one go (separate with commas, no spaces):

.. code-block:: console

   $ torchgeo-bench run model=my_model \
       dataset.names=[m-eurosat,m-so2sat,m-bigearthnet,m-brick-kiln,m-forestnet,m-pv4ger]

For V2 classification and segmentation datasets:

.. code-block:: console

   $ torchgeo-bench run model=my_model \
       dataset.names=[benv2,treesatai,so2sat,forestnet]
   $ torchgeo-bench run model=my_model \
       dataset.names=[burn_scars,caffe,cloudsen12,dynamic_earthnet]

**Sensor-coverage guidance**: skip datasets whose sensor modality your model
was not trained on.  Document skipped datasets as inline comments in your
notes — e.g. "``m-forestnet`` skipped: model trained on S2 only; Landsat
not supported".

Skip the (slow) linear probe and reduce bootstrap samples for a quick trial:

.. code-block:: console

   $ torchgeo-bench run model=my_model dataset.names=[m-eurosat] \
       eval.skip_linear=true eval.bootstrap=100

If a run is interrupted, resume from where it left off:

.. code-block:: console

   $ torchgeo-bench run model=my_model resume=true

.. _eval-results:

Interpreting results
--------------------

Results are written to ``results/all_results.csv`` as they are computed.
Each row is one ``(dataset, method, model, config)`` measurement.  The key
columns are:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Meaning
   * - ``dataset``
     - Dataset CLI name (e.g. ``m-eurosat``).
   * - ``method``
     - ``knn5`` (KNN-5) or ``linear`` (L-BFGS logistic regression).
   * - ``metric_name``
     - ``accuracy`` (single-label) or ``micro_mAP`` (multi-label).
   * - ``metric_value``
     - Point estimate on the test split.
   * - ``ci_lower`` / ``ci_upper``
     - 95 % bootstrap confidence interval bounds (stratified by class).
       The default uses 1 000 resamples; tune with ``eval.bootstrap=N``.
   * - ``feature_dim``
     - Embedding dimension from your backbone.
   * - ``partition``
     - GeoBench V1 partition name (``default`` for V2).
   * - ``bands``
     - Which input channels were used (``rgb``, ``all``, or a sorted
       comma-joined list of band names).

Read the CSV directly with pandas:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv("results/all_results.csv")
   my_model = df[df["name"] == "my_model"]
   print(my_model[["dataset", "method", "metric_value", "ci_lower", "ci_upper"]])

For the full column reference, see :doc:`results-format`.

.. _eval-cite:

Citing torchgeo-bench
---------------------

If you use ``torchgeo-bench`` in a paper or report, please cite:

.. code-block:: bibtex

   @software{torchgeo_bench,
     author       = {torchgeo-bench contributors},
     title        = {{torchgeo-bench}: Frozen geospatial foundation model benchmark},
     year         = {2024},
     url          = {https://github.com/torchgeo/torchgeo-bench},
   }
