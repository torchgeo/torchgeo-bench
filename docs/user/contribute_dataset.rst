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

   from torchgeo_bench.datasets.base import BenchDataset, BandSpec

   class MyDataset(BenchDataset):
       name = "my_dataset"
       task = "classification"        # or "segmentation"
       num_classes = 10
       bands: list[BandSpec] = [...]
       split_sizes = {"train": 5000, "val": 1000, "test": 2000}

       def get_dataset(self, split: str, bands) -> torch.utils.data.Dataset:
           ...  # return a Dataset yielding (image_tensor, label) pairs

Required class-level attributes:

* ``name`` — unique string identifier used by the Hydra registry and CLI
* ``task`` — ``"classification"`` or ``"segmentation"``
* ``num_classes`` — integer label count
* ``bands`` — list of :class:`~torchgeo_bench.datasets.base.BandSpec` objects
  supplying per-channel sensor / wavelength / normalisation stats
* ``split_sizes`` — dict with ``train``, ``val``, and ``test`` keys

The ``get_dataset`` method must accept ``split`` (``"train"``, ``"val"``, or
``"test"``) and ``bands`` (the subset of bands requested by the model), and
return a :class:`torch.utils.data.Dataset` whose ``__getitem__`` yields
``(image_tensor, label)`` pairs.

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

**1. Export the class** from :file:`src/torchgeo_bench/datasets/__init__.py`:

.. code-block:: python

   from .my_dataset import MyDataset

**2. Add a Hydra dataset config YAML** under
:file:`src/torchgeo_bench/conf/dataset/` named after your dataset
(e.g. :file:`my_dataset.yaml`):

.. code-block:: yaml

   # @package _global_
   defaults:
     - base_dataset

   dataset:
     name: my_dataset
     num_classes: 10
     task: classification

Adjust keys as needed; see existing configs in that directory for reference.

Run the smoke test
------------------

With the dataset on disk, run a quick benchmark to verify the dataset loads
and produces sensible results:

.. code-block:: console

   $ torchgeo-bench run model=timm/resnet50 dataset.names=[my_dataset] \
       eval.skip_linear=true eval.bootstrap=10

Once results look sensible, follow the PR workflow described in
:doc:`contribute_model` to open a pull request.
