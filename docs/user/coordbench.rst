CoordBench location encoders
============================

CoordBench evaluates frozen location encoders without imagery. Each benchmark
maps a point ``(longitude, latitude[, year])`` to one or more downstream labels.
The runner embeds each point once, then fits KNN or ridge-linear probes under
random, spatial-block, or official held-out splits.

The normalized benchmark tables stream from the
`taylor-geospatial/coordbench <https://huggingface.co/datasets/taylor-geospatial/coordbench>`_
dataset on Hugging Face. There is no separate download command and no local
``data/`` layout to prepare.

First run
---------

The base installation includes the dependency-free sine/cosine baseline. This
small CPU example evaluates one regression benchmark with two random folds:

.. code-block:: console

   $ torchgeo-bench run \
       mode=coord \
       model=sincos \
       coord.names=california_housing \
       coord.methods=[linear] \
       coord.folds=2 \
       device=cpu \
       coord.output=results/coordbench_quickstart.csv

Results are appended to ``coord.output``. Set ``resume=true`` to skip rows that
already match ``(dataset, task, method, model, split)``.

Included encoders
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 60

   * - Model preset
     - Installation
     - Description
   * - ``sincos``
     - base
     - Four-dimensional periodic coordinate baseline.
   * - ``mind``
     - base
     - 64-dimensional MIND Matryoshka prefix.
   * - ``mind-small``
     - base
     - 128-dimensional distilled MIND student.
   * - ``climplicit``
     - ``coordbench``
     - Climplicit climate-specialist encoder.
   * - ``geoclip``
     - ``coordbench``
     - GeoCLIP Equal-Earth/RFF encoder.
   * - ``satclip``
     - ``coordbench``
     - SatCLIP spherical-harmonic encoder.
   * - ``sinr``
     - ``coordbench``
     - SINR species-distribution encoder.
   * - ``xyz``
     - base
     - Unit-sphere XYZ position encoding.
   * - ``nerf``
     - base
     - NeRF Fourier position encoding.
   * - ``spherical-harmonics``
     - base
     - Compact real spherical-harmonic encoding.

Install the optional reference encoders from PyPI with:

.. code-block:: console

   $ pip install "torchgeo-bench[coordbench]"

For example, run SatCLIP on its five downstream benchmarks with random and
spatial-block cross-validation:

.. code-block:: console

   $ torchgeo-bench run \
       mode=coord \
       model=satclip \
       coord.names=satclip \
       coord.split=both

Benchmarks and probes
---------------------

``coord.names`` accepts ``all``, a comma-separated list, a YAML list, a family
name, or an individual benchmark name. Available families are ``pdfm``,
``air_temp``, ``california_housing``, ``satclip``, ``sustainbench``,
``better_together``, ``cdc_places``, ``usavars``, ``country``, ``ecoregions``,
``worldclim``, ``soilgrids``, and ``deepmind``.

Classification tasks report accuracy and support ``knn`` and ``linear``.
Regression tasks report R2 and use ``linear``; requested KNN rows are skipped
because this track does not define a KNN regressor. ``coord.split`` controls the
holdout:

``random``
   Seeded k-fold cross-validation.
``spatial``
   K-fold cross-validation over geographic grid cells. ``coord.cell_deg`` sets
   the cell width in degrees.
``both``
   Run both cross-validation protocols. Benchmarks with an official test mask
   use that fixed holdout once instead.

The full block is:

.. code-block:: yaml

   coord:
     output: results/coordbench_results.csv
     names: all
     methods: [knn, linear]
     split: random
     folds: 5
     cell_deg: 10.0
     knn_k: 5
     knn_device: cpu

Spatial-prior baselines
-----------------------

The coordinate encoders are evaluated as frozen representations. CoordBench
also has a separate ``coord-prior`` mode for label-informed baselines. These
methods fit on the training coordinates and labels, predict the held-out
coordinates, and write to a separate file:

.. code-block:: console

   $ torchgeo-bench run \
       mode=coord-prior \
       coord_prior.names=satclip \
       coord_prior.methods=[uniform,frequency,grid,nearest,kde] \
       coord_prior.split=both \
       coord_prior.output=results/coordbench_priors.csv

Available methods are ``uniform``, ``frequency``, ``grid``, ``nearest``, and
``kde``. They currently apply to classification tasks only; regression tasks
are skipped. The prior output should be analyzed separately from frozen
location-encoder results because these methods use task labels during fitting.

External released weights are often dataset-specific classifiers whose
checkpoints also encode the original task head and, for some families, anchor
locations. They are not silently reinterpreted as universal CoordBench models.
The coordinate-only pretrained encoders remain available through the
``coordbench`` extra. Retrieval-augmented models that require an external
database are intentionally outside this apples-to-apples encoder track.

Add a location encoder
----------------------

A custom encoder subclasses :class:`~torchgeo_bench.coordbench.LocationEncoder`
and implements ``_encode(lon, lat, year)``. The method receives one batch of
NumPy arrays and returns a finite ``(N, D)`` feature matrix.

The repository includes a complete Fourier-feature example in
:file:`examples/coordbench_location_encoder.py`. Run it from the repository
root by replacing the built-in ``sincos`` target through a config override:

.. code-block:: console

   $ PYTHONPATH=. uv run torchgeo-bench run \
       mode=coord \
       model=sincos \
       model._target_=examples.coordbench_location_encoder.FourierLocationEncoder \
       model.name=fourier \
       +model.num_frequencies=8 \
       coord.names=california_housing \
       coord.methods=[linear] \
       coord.folds=2 \
       device=cpu \
       coord.output=results/fourier_coordbench.csv

For a reusable integration, place the class in an installed package and add a
model YAML under :file:`src/torchgeo_bench/conf/model/` with its fully qualified
``_target_``. See :doc:`/api/coordbench` for the public classes and probe
functions.

Aggregate results
-----------------

The leaderboard helper groups scores by task family and reports mean rank for
random and spatial holdouts:

.. code-block:: console

   $ python -m torchgeo_bench.coordbench.leaderboard \
       results/coordbench_results.csv --method linear
