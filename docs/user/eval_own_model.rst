Evaluate your own model (Stage 1)
==================================

This guide demonstrates how you can benchmark any frozen pretrained geospatial model against
the included benchmark datasets. If you want to contribute a new open-source model with available weights such that the broader community can easily access your model, see
:doc:`contribute_model` (Stage 2).

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

   $ uv sync --extra newextra

You can check how to download one or more dataset for evaluation in the :doc:`datasets` guide.

.. _eval-implement:

Implement your model
--------------------

We provide a template file to give you a general setup and fill in the gaps that are unique to your model and ensure
that each of those parts will be used correctly in the benchmark pipeline. Copy the template file to your working directory and fill in the ``TODO``
sections:

.. code-block:: console

   $ cp src/torchgeo_bench/models/contrib_template.py ./new_model.py

The template is a single class ``NewModel``. One of the most important parts is carefully configuring the correct
normalization-choice block in ``__init__``. Change the one ``normalization=``
line to match the configurateion of your backbone.

**Normalization strategy decision table**

Pick the strategy that matches how your backbone was trained:

.. list-table::
   :header-rows: 1
   :widths: 20 42 38

   * - Strategy
     - When to use — in-repo examples
     - How to set it
   * - ``bandspec_zscore``
     - The framework z-scores each channel from
       the dataset's BandSpec statistics, with the goal of producing ~N(0, 1) inputs regardless
       of source sensor unit.

       *In-repo examples:* ScaleMAE, Satlas Swin, EarthLoc, SAM3,
       all timm ImageNet models (ResNet-50, ViT-B/16, ConvNeXt, …), RCF.
     - Default; leave the ``normalization=`` line as-is.
   * - ``identity``
     - Your backbone ships its own normalizer and must receive raw sensor
       values — applying a second normalization on top would corrupt the
       inputs.

       *In-repo example:* OlmoEarth — its internal ``Normalizer`` consumes
       raw DN/reflectance directly and auto-detects the sensor scale.  See
       :class:`~torchgeo_bench.models.OlmoEarthBenchModel` and
       :file:`src/torchgeo_bench/models/olmoearth.py` for the pattern.
     - Change to ``normalization="identity"`` in the ``super().__init__`` call. which will skip
       the dataset normalization in the pipeline
   * - ``model_native``
     - The exact pretraining input scale is published and you can declare it
       explicitly.  The framework converts the dataset's sensor unit to the
       backbone's expected unit, then applies any declared per-channel
       mean/std.

       *In-repo examples:* Prithvi-EO (``expected_input_unit = S2_DN``),
       Clay v1.5 and TerraMind (``expected_input_unit = REFLECTANCE_0_1``),
       CROMA (``expected_input_unit = REFLECTANCE_0_1``).  See
       ``TerraTorchPrithviBench`` in :file:`src/torchgeo_bench/models/terratorch_models.py`
       and :class:`~torchgeo_bench.models.TimmPatchBenchModel` for the pattern.
     - Set ``expected_input_unit``, ``pretrain_mean``, and ``pretrain_std``
       as class attributes *before* calling ``super().__init__(bands=bands)``.

For the full list of available strategies and their exact semantics, see
:file:`src/torchgeo_bench/models/_normalization.py`.

Accessing band metadata
^^^^^^^^^^^^^^^^^^^^^^^

The template shows ``backbone(images)`` as the minimal forward call, but many
models need more than raw pixels — for example a wavelength list for
band-agnostic ViTs, or sensor-conditional routing.

The framework makes this straightforward.  The pipeline **reinstantiates your
class once per dataset**, so the ``bands`` argument passed to ``__init__``
always reflects exactly the channels being loaded for that run.  Every
:class:`~torchgeo_bench.datasets.base.BandSpec` in that list carries the
dataset-level metadata that is available:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``wavelength_um``
     - Centre wavelength in micrometres (``None`` for non-optical bands such as
       SAR backscatter or DEM elevation).  Use this to drive wavelength-aware
       embeddings (e.g. DOFA).
   * - ``sensor``
     - Sensor family string — ``"s2"``, ``"landsat"``, ``"sar"``, ``"aerial"``,
       ``"planet"``, ``"worldview"``.  Use this for sensor-conditional routing
       or to detect unsupported modalities at construction time.
   * - ``name``
     - Canonical short band name — ``"red"``, ``"nir"``, ``"vv"``, ``"b02"``.
       Use this when your backbone expects bands in a named order.

The pattern is: extract what you need from ``bands`` in ``__init__`` and store
it as an instance attribute, then use it in ``_forward_patch_features``:

.. code-block:: python

   from torchgeo_bench.datasets.base import BandSpec
   from torchgeo_bench.models.interface import BenchModel

   class NewModel(BenchModel):
       def __init__(self, bands: list[BandSpec], **kwargs) -> None:
           super().__init__(bands=bands, normalization="bandspec_zscore")

           # The runner reinstantiates this class once per dataset, so these
           # attributes are always current for the channels being loaded.
           self.wavelengths = [b.wavelength_um for b in bands]  # None for SAR/DEM
           self.sensors = [b.sensor for b in bands]             # e.g. "s2", "landsat"
           self.band_names = [b.name for b in bands]            # e.g. "red", "nir"

           self.backbone = ...  # your backbone here

       def _forward_patch_features(self, images, _bboxes=None):
           # Pass the cached metadata alongside the image tensor.
           return self.backbone(images, wavelengths=self.wavelengths)

For a complete example see ``TorchGeoDOFABench`` in
:file:`src/torchgeo_bench/models/torchgeo_models.py`, which reads
``wavelength_um`` from each ``BandSpec`` at construction and passes the
resulting list to ``backbone.forward_features(images, wavelengths=...)``.

.. _eval-hydra-config:

Create a Hydra config
---------------------

Create a model YAML file at :file:`src/torchgeo_bench/conf/model/new_model.yaml`.
The only required key is ``_target_``, which must point to your class:

.. code-block:: yaml

   # src/torchgeo_bench/conf/model/new_model.yaml
   _target_: new_model.NewModel    # dotted import path to your class
   pretrained: true
   name: new_model                 # human-readable label in the results CSV

   # Add any kwargs your __init__ accepts (except `bands` — see note below).
   # embed_dim: 768
   # checkpoint: path/to/weights.pt

.. note::

   **Do not put** ``bands`` **in the YAML.**  The pipeline reads the current
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

Pass your config name as ``model=new_model`` and any combination of dataset
names to the ``run`` subcommand (see :doc:`datasets` for the full list of
available names):

.. code-block:: console

   $ torchgeo-bench run model=new_model dataset.names=[m-eurosat]
   $ torchgeo-bench run model=new_model \
       dataset.names=[m-eurosat,m-bigearthnet,benv2,burn_scars]

Skip the (slow) linear probe and reduce bootstrap samples for a quick trial:

.. code-block:: console

   $ torchgeo-bench run model=new_model dataset.names=[m-eurosat] \
       eval.skip_linear=true eval.bootstrap=100

To write results to a dedicated file instead of the shared
``results/all_results.csv``, pass ``output=``:

.. code-block:: console

   $ torchgeo-bench run model=new_model \
       dataset.names=[m-eurosat,m-so2sat] \
       output=results/new_model_results.csv

The ``resume=true`` flag respects whatever ``output=`` is set to, so an
interrupted run can be continued against the same file:

.. code-block:: console

   $ torchgeo-bench run model=new_model output=results/new_model_results.csv resume=true

.. _eval-results:

Results
-------

Results are written to ``results/all_results.csv`` by default, or to the
path set via ``output=`` (see above).
For the full column reference and how to read the CSV, see :doc:`results-format`.
