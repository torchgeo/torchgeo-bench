torchgeo_bench.datasets
=======================

.. module:: torchgeo_bench.datasets

Every benchmark dataset is a subclass of :class:`BenchDataset` that declares
its metadata (bands, number of classes, task type, split sizes) and knows how
to produce a PyTorch :class:`~torch.utils.data.Dataset` for each split.
Datasets are registered automatically on import so that
:func:`get_bench_dataset_class` can resolve them by their CLI name
(e.g. ``"m-eurosat"`` or ``"benv2"``).

Base classes
------------

.. autoclass:: BenchDataset
.. autoclass:: BandSpec

``BenchDataset.resolve_band_specs()`` translates ``rgb``, ``all``, or an ordered iterable of band names into the dataset's existing ``BandSpec`` objects without loading samples. RGB selection follows each dataset's metadata, including grayscale and SAR datasets.

Loading API
-----------

``load_split(name, split, ...)`` loads only the requested split and returns a
``LoadedSplit``. Its ``dataset`` is the PyTorch Dataset; ``bands`` exposes the
original ordered ``BandSpec`` objects used by the source loader and model.
``input`` describes the selection and actual image layout: CHW for ordinary
or single-step inputs, TCHW for explicit multi-step PASTIS inputs. Identity,
partition, task, class count, multilabel status, and target key accompany the data.

Callers own DataLoaders, including shuffle, workers, pinning, and seeds:

.. code-block:: python

   from torch.utils.data import DataLoader
   from torchgeo_bench.datasets import load_split

   train = load_split("m-eurosat", "train", bands="rgb", image_size=64)
   loader = DataLoader(train.dataset, batch_size=32, shuffle=True, num_workers=0)
   # Supply list(train.bands) to build_model; do not select model bands again.

Request validation/test explicitly when needed. The image runner applies a
non-default partition only to training; a direct caller can request a supported
partition on any individual split. Unsupported partitions or temporal options,
invalid splits, and invalid band selections fail before source construction.
There are no automatic downloads or sample probes in ``load_split``.
Runtime split objects are not configuration or resume-hash payloads.

.. autofunction:: load_split
.. autoclass:: LoadedSplit
.. autoclass:: ResolvedInput
.. autofunction:: get_bench_dataset_class
.. autofunction:: list_datasets

GeoBench V1 (classification)
----------------------------

V1 datasets use the ``m-`` prefix on the command line.  They wrap the original
JSON-metadata shards or custom JSON-metadata HDF5 and expose the standard ``train``/``val``/``test``
splits plus alternative partitions where available.

.. autoclass:: MBigEarthNet
.. autoclass:: MBrickKiln
.. autoclass:: MEurosat
.. autoclass:: MForestnet
.. autoclass:: MPv4ger
.. autoclass:: MSo2Sat

GeoBench V2 — classification
----------------------------

.. autoclass:: BENV2
.. autoclass:: Forestnet
.. autoclass:: So2Sat
.. autoclass:: TreeSatAI

GeoBench V2 — segmentation
--------------------------

.. autoclass:: BurnScars
.. autoclass:: CaFFe
.. autoclass:: CloudSEN12
.. autoclass:: DynamicEarthNet
.. autoclass:: FLAIR2
.. autoclass:: FieldsOfTheWorld
.. autoclass:: KuroSiwo
.. autoclass:: PASTIS
.. autoclass:: SpaceNet2
.. autoclass:: SpaceNet7

torchgeo wrappers
-----------------

.. autoclass:: EuroSAT

.. autoclass:: EuroSATSpatial

.. autoclass:: RESISC45
