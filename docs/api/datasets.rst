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

Loading API
-----------

.. autofunction:: get_datasets
.. autofunction:: get_bench_dataset_class
.. autofunction:: list_datasets

GeoBench V1 (classification)
----------------------------

V1 datasets use the ``m-`` prefix on the command line.  They wrap the original
GeoBench HDF5 distributions and expose the standard ``train``/``val``/``test``
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
