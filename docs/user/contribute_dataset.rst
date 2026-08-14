Add a Dataset
=============

This page explains how to wire a new geospatial dataset into torchgeo-bench
so that any registered model can be evaluated on it automatically.

Prerequisites
-------------

Clone the repository and install the development dependencies:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ conda activate torchgeo-bench
   $ uv sync --extra dev

This is the same setup used in :doc:`eval_own_model`.  Download the dataset
files so you can test loading locally:

.. code-block:: console

   $ torchgeo-bench download <dataset_name>

Implement BenchDataset
----------------------

Create a new module under :file:`src/torchgeo_bench/datasets/` and subclass
:class:`~torchgeo_bench.datasets.base.BenchDataset`:

.. code-block:: python

   from pathlib import Path
   from collections.abc import Callable

   from torchgeo_bench.datasets.base import BenchDataset, BandSpec

   class MyDataset(BenchDataset):
       name = "my_dataset"
       task = "classification"        # or "segmentation"
       num_classes = 10
       bands: list[BandSpec] = [...]
       rgb_bands = ["red", "green", "blue"]
       split_sizes = {"train": 5000, "val": 1000, "test": 2000}

       @classmethod
       def data_root(cls) -> Path:
           return Path("data/my_dataset")

       def get_dataset(
           self,
           split: str,
           *,
           partition: str = "default",
           bands: tuple[str, ...] | None = None,
           transform: Callable | None = None,
       ) -> torch.utils.data.Dataset:
           ...  # return a Dataset yielding {"image": tensor, "label": tensor} samples

Required class-level attributes:

* ``name`` — unique string identifier used by the dataset registry and CLI
* ``task`` — ``"classification"`` or ``"segmentation"``
* ``num_classes`` — integer label count
* ``bands`` — list of :class:`~torchgeo_bench.datasets.BandSpec` objects
  supplying per-channel sensor / wavelength / normalisation stats
* ``rgb_bands`` — short names of the bands used in RGB-only mode
* ``split_sizes`` — dict with ``train``, ``val``, and ``test`` keys

The ``get_dataset`` method takes ``split`` (``"train"``, ``"val"``, or
``"test"``) plus the keyword-only arguments ``partition``, ``bands`` (the
subset of bands requested by the model), and ``transform``.  It returns a
:class:`torch.utils.data.Dataset` whose ``__getitem__`` yields **dict**
samples — ``{"image": tensor, "label": tensor}`` for classification, with
``"mask"`` in place of ``"label"`` for segmentation.  Datasets always emit
raw float32 values; normalization is the model's job.

If you inherit from ``_V1Dataset`` or ``_V2Dataset``, both ``data_root`` and
``get_dataset`` are already implemented — your subclass is pure metadata.  See
:file:`src/torchgeo_bench/datasets/m_eurosat.py` for a minimal example.

.. note::

   **V1 vs V2 loader patterns.** V1 datasets (``m-`` prefix) read images
   directly from HDF5 files via
   :class:`~torchgeo_bench.datasets.geobench_v1._V1Dataset`.  V2 datasets use
   torchgeo dataset classes as the underlying loader and inherit from
   :class:`~torchgeo_bench.datasets.geobench_v2._V2Dataset`.  When adding a
   genuinely new dataset, prefer the V2 torchgeo pattern so the loader can
   participate in torchgeo's transform pipeline.

Register and configure
----------------------

**1. Register the class** by adding an entry to ``_REGISTRY_SPEC`` in
:file:`src/torchgeo_bench/datasets/loading.py`, mapping the dataset name to
its ``(submodule, class_name)``:

.. code-block:: python

   _REGISTRY_SPEC: dict[str, tuple[str, str]] = {
       ...,
       "my_dataset": ("my_dataset", "MyDataset"),
   }

This is the step that actually makes the dataset available — it backs
``get_bench_dataset_class``, :func:`~torchgeo_bench.datasets.list_datasets`,
and the CLI.  There is no per-dataset Hydra config; datasets are selected by
name on the command line. The registry is kept as module/class-name strings
rather than imported classes so that importing ``loading`` stays cheap;
``get_bench_dataset_class`` imports only the one module it needs.

**2. Export the class** from :file:`src/torchgeo_bench/datasets/__init__.py`
by adding an ``__all__`` entry and a matching ``_LAZY_CLASSES`` mapping:

.. code-block:: python

   __all__ = [
       ...,
       "MyDataset",
   ]

   _LAZY_CLASSES: dict[str, str] = {
       ...,
       "MyDataset": "my_dataset",
   }

Individual dataset classes load lazily through module ``__getattr__`` so that
``import torchgeo_bench.datasets`` — and CLI startup — stays fast.  Do not add
an eager ``from .my_dataset import MyDataset`` import; register the name in
``_LAZY_CLASSES`` instead.

**3. For a GeoBench V2 dataset**, also add the name → upstream class mapping to
``_V2_REGISTRY`` in :file:`src/torchgeo_bench/datasets/geobench_v2.py` and to
``DEFAULT_V2_DATASETS`` in :file:`src/torchgeo_bench/download.py`.

**4. Add the expected split sizes** to ``EXPECTED_SIZES`` in
:file:`tests/test_split_sizes.py`.

Run the smoke test
------------------

With the dataset on disk, run a quick benchmark to verify the dataset loads
and produces sensible results:

.. code-block:: console

   $ torchgeo-bench run -m timm/resnet50 -d my_dataset --skip-linear --bootstrap 10

Add the geographic metadata
---------------------------

Every registered dataset carries a record in the committed geographic store
under :file:`docs/_static/_dataset_geography/`, which drives the spatial
coverage map.  Generate yours:

.. code-block:: console

   $ python experiments/scripts/extract_dataset_geography.py --dataset my_dataset

Commit the resulting :file:`docs/_static/_dataset_geography/my_dataset.json`
together with the regenerated :file:`index.json`.

:file:`tests/test_geography.py` fails when a registered dataset has no record,
so this step is not optional — but "no coordinates" is a perfectly valid
answer.  If the dataset genuinely carries no geolocation, add it to ``NO_GEO``
in :file:`src/torchgeo_bench/geography.py` with the reason you verified, and
the map will disclose the gap rather than quietly omit the dataset.

Coordinates are read from the raw files (V2 ``.tortilla`` metadata columns, V1
HDF5 affine transforms).  A dataset stored in some other layout needs a branch
in ``extract_geography``.

Once results look sensible, follow the PR workflow described in
:doc:`contribute_model` to open a pull request.
