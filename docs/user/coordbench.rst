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

   $ torchgeo-bench coord --model sincos --dataset california_housing \
       --methods linear --folds 2 --device cpu \
       --output results/coordbench_quickstart.csv

Results are appended to ``output.file`` (``--output``). Add ``--resume`` to
skip rows that already match ``(dataset, task, method, model_name, split)``.
Unlike image-run resume, this key does not include a configuration hash;
use a separate output file when changing encoder kwargs or evaluation
settings that are not part of the key.

``python -m torchgeo_bench coord`` and ``python -m torchgeo_bench.cli coord``
are equivalent entry points. The former ``run mode=coord`` and dotted
``key=value`` overrides are no longer supported.

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

   $ torchgeo-bench coord --model satclip --dataset satclip --split both

Benchmarks and probes
---------------------

``--dataset`` accepts one or more family or benchmark names, for example
``--dataset pdfm satclip``. The corresponding YAML field is a list:
``datasets: [pdfm, satclip]``. Use ``--dataset all`` or ``datasets: [all]``
to select the full suite; do not combine ``all`` with other names.
Available families are ``pdfm``,
``air_temp``, ``california_housing``, ``satclip``, ``sustainbench``,
``better_together``, ``cdc_places``, ``usavars``, ``country``, ``ecoregions``,
``worldclim``, ``soilgrids``, and ``deepmind``.

Classification tasks report accuracy and support ``knn`` and ``linear``.
Regression tasks report R2 and use ``linear``; requested KNN rows are skipped
because this track does not define a KNN regressor. ``evaluation.split``
(``--split``) controls the
holdout:

``random``
   Seeded k-fold cross-validation.
``spatial``
   K-fold cross-validation over geographic grid cells. ``evaluation.cell_deg``
   (``--cell-deg``) sets the cell width in degrees.
``both``
   Run both cross-validation protocols. Benchmarks with an official test mask
   use that fixed holdout once instead.

Benchmarks with an official test mask use that fixed holdout regardless of the
requested cross-validation protocol; ``both`` does not duplicate it. CSV split
labels are ``random``, ``spatial``, or ``official``. Cross-validation intervals
report mean plus/minus the standard deviation across folds, not image-style
bootstrapped confidence intervals.

YAML configuration
------------------

Pass a file to ``torchgeo-bench coord --config my-coord.yaml``:

.. code-block:: yaml

   schema_version: 1
   model:
     name: sincos
   datasets: [california_housing]
   evaluation:
     methods: [knn, linear]
     split: random
     folds: 5
     cell_deg: 10.0
     knn_k: 5
     knn_device: cpu
   runtime:
     device: cpu
     seed: 0
   output:
     file: results/coordbench_results.csv
     resume: false

Explicit flags override YAML, including ``--no-resume``.
``--methods knn``, ``--methods linear``, and ``--methods knn linear`` select
probes. ``--folds`` must be at least two. Other evaluation flags are
``--cell-deg``, ``--knn-k``, and ``--knn-device``. ``--device`` selects the
encoder/linear-probe device, and ``--seed`` controls reproducibility.

``--dry-run`` validates and prints YAML without loading encoder weights or
remote tables:

.. code-block:: console

   $ torchgeo-bench coord --config examples/coord-run.yaml --dry-run

The coordinate schema deliberately has no image ``input``, ``segmentation``,
or ``classification`` sections. See :doc:`/api/coordbench` for
``CoordConfig`` and the Python loader.

Add a location encoder
----------------------

A custom encoder subclasses :class:`~torchgeo_bench.coordbench.LocationEncoder`
and implements ``_encode(lon, lat, year)``. The method receives one batch of
NumPy arrays and returns a finite ``(N, D)`` feature matrix.

The repository includes a complete Fourier-feature example in
:file:`examples/coordbench_location_encoder.py`. Run it from the repository
root using :file:`examples/coord-run.yaml`:

.. code-block:: yaml

   model:
     name: fourier
     target: examples.coordbench_location_encoder.FourierLocationEncoder
     kwargs:
       num_frequencies: 8
   datasets: [california_housing]
   evaluation:
     methods: [linear]
     folds: 2
   runtime:
     device: cpu
   output:
     file: results/fourier_coordbench.csv

.. code-block:: console

   $ PYTHONPATH=. uv run torchgeo-bench coord --config examples/coord-run.yaml

For a reusable integration, place the class in an installed package and add a
model preset under :file:`src/torchgeo_bench/conf/model/`:

.. code-block:: yaml

   name: fourier
   target: my_package.FourierLocationEncoder
   track: coord
   kwargs:
     num_frequencies: 8

``track`` is preset metadata, not a constructor argument. The runner passes
the execution device explicitly when constructing the encoder. See
:doc:`configuration` for preset resolution and :doc:`/api/coordbench` for
the public classes and probe functions.

Aggregate results
-----------------

The leaderboard helper groups scores by task family and reports mean rank for
random and spatial holdouts:

.. code-block:: console

   $ python -m torchgeo_bench.coordbench.leaderboard \
       results/coordbench_results.csv --method linear
