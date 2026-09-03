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

   $ torchgeo-bench coord \
       --model sincos \
       --names california_housing \
       --methods linear \
       --folds 2 \
       --device cpu \
       --output results/coordbench_quickstart.csv

Results are appended to ``--output`` (default: ``results/coordbench_results.csv``).
Pass ``--resume`` to skip rows that already match
``(dataset, task, method, model, split)``.

Included encoders
-----------------

================  ==============  =============================================
Model preset      Installation    Description
================  ==============  =============================================
``sincos``        base            Four-dimensional periodic coordinate baseline.
``mind``          base            64-dimensional MIND Matryoshka prefix.
``mind-small``    base            128-dimensional distilled MIND student.
``climplicit``    ``coordbench``  Climplicit climate-specialist encoder.
``geoclip``       ``coordbench``  GeoCLIP Equal-Earth/RFF encoder.
``satclip``       ``coordbench``  SatCLIP spherical-harmonic encoder.
``sinr``          ``coordbench``  SINR species-distribution encoder.
================  ==============  =============================================

Install the optional reference encoders from PyPI with:

.. code-block:: console

   $ pip install "torchgeo-bench[coordbench]"

For example, run SatCLIP on its five downstream benchmarks with random and
spatial-block cross-validation:

.. code-block:: console

   $ torchgeo-bench coord \
       --model satclip \
       --names satclip \
       --split both

Benchmarks and probes
---------------------

``coord.names`` accepts ``all``, a comma-separated list, a YAML list, a family
name, or an individual benchmark name. Available families are ``pdfm``,
``air_temp``, ``california_housing``, ``satclip``, ``sustainbench``,
``better_together``, ``cdc_places``, ``usavars``, ``country``, ``ecoregions``,
``worldclim``, ``soilgrids``, and ``deepmind``.

Classification tasks report accuracy and support ``knn`` and ``linear``.
Regression tasks report R2 and use ``linear``; requested KNN rows are skipped
because this track does not define a KNN regressor. ``--split`` controls the
holdout:

``random``
   Seeded k-fold cross-validation.
``spatial``
   K-fold cross-validation over geographic grid cells. ``--cell-deg`` sets
   the cell width in degrees.
``both``
   Run both cross-validation protocols. Benchmarks with an official test mask
   use that fixed holdout once instead.

``torchgeo-bench coord --help`` lists every flag with its default
(``--output``, ``--names``, ``--methods``, ``--split``, ``--folds``,
``--cell-deg``, ``--knn-k``, ``--knn-device``, plus the common ``--model``,
``--device``, ``--resume``, ``--seed``, ``--config``, ``--print-config``).

Add a location encoder
----------------------

A custom encoder subclasses :class:`~torchgeo_bench.coordbench.LocationEncoder`
and implements ``_encode(lon, lat, year)``. The method receives one batch of
NumPy arrays and returns a finite ``(N, D)`` feature matrix.

The repository includes a complete Fourier-feature example in
:file:`examples/coordbench_location_encoder.py`. There is no ``key=value``
override for an arbitrary ``_target_``, so trying it out means adding a small
preset YAML that points at the example class, e.g.
:file:`src/torchgeo_bench/conf/model/fourier.yaml`:

.. code-block:: yaml

   _target_: examples.coordbench_location_encoder.FourierLocationEncoder
   name: fourier
   num_frequencies: 8

then running it from the repository root with ``--model fourier``:

.. code-block:: console

   $ PYTHONPATH=. uv run torchgeo-bench coord \
       --model fourier \
       --names california_housing \
       --methods linear \
       --folds 2 \
       --device cpu \
       --output results/fourier_coordbench.csv

For a reusable integration, place the class in an installed package and add a
model preset under :file:`src/torchgeo_bench/conf/model/` with its fully
qualified ``_target_``. See :doc:`/api/coordbench` for the public classes and
probe functions.

Aggregate results
-----------------

The leaderboard helper groups scores by task family and reports mean rank for
random and spatial holdouts:

.. code-block:: console

   $ python -m torchgeo_bench.coordbench.leaderboard \
       results/coordbench_results.csv --method linear
