torchgeo_bench.models
=====================

.. module:: torchgeo_bench.models

This module provides the abstract :class:`BenchModel` interface and a
collection of concrete backbones that can be benchmarked across the
:mod:`torchgeo_bench.datasets` registry.

Interface
---------

.. autoclass:: BenchModel

Backbones
---------

Random Convolutional Features
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: RCFBench

Image statistics baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: ImageStatsBench

timm encoders
^^^^^^^^^^^^^

.. autoclass:: TimmPatchBenchModel

torchgeo encoders
^^^^^^^^^^^^^^^^^

.. autoclass:: TorchGeoResNetBench
.. autoclass:: TorchGeoSwinBench
.. autoclass:: TorchGeoScaleMAEBench
.. autoclass:: TorchGeoDOFABench
.. autoclass:: TorchGeoEarthLocBench

OlmoEarth
^^^^^^^^^

.. autoclass:: OlmoEarthBenchModel

SAM 3
^^^^^

.. autoclass:: SAM3Encoder

Segmentation heads
------------------

These heads attach to a frozen :class:`BenchModel` backbone to produce
dense per-pixel predictions.  See :class:`~torchgeo_bench.segmentation_probe.SegmentationProbe`
for the wiring layer, and :doc:`/user/segmentation-layers` for the
verified ``eval.segmentation.layers`` values for each supported timm
backbone family.

.. autoclass:: LinearHead
.. autoclass:: ConvBlockHead
.. autoclass:: FPNHead
.. autoclass:: DPTHead
