Models
======

This page is the operator-facing tour of the model backbones bundled
with ``torchgeo-bench``: which presets exist, how to invoke them, and
how to add a new one.  For the abstract base class and the full class
reference, see :doc:`/api/models`.

.. _model-presets:

Available presets
-----------------

Every preset under :file:`src/torchgeo_bench/conf/model/` becomes a
``model=…`` selector for the ``run`` subcommand.  A preset's ``_target_``
field resolves to a class re-exported from :mod:`torchgeo_bench.models`.

Random Convolutional Features (RCF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.RCFBench`. Gaussian or empirical random
features in the spirit of MOSAIKS.

.. code-block:: console

   $ torchgeo-bench run model=rcf
   $ torchgeo-bench run model=rcf model.mode=empirical model.features=1024

Image statistics baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.ImageStatsBench`. A trivial baseline that
returns per-channel mean / std as the feature vector.

.. code-block:: console

   $ torchgeo-bench run model=imagestats

timm — ImageNet-pretrained CNNs and ViTs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.TimmPatchBenchModel`.  Configs under
:file:`src/torchgeo_bench/conf/model/timm/` cover ResNet, ConvNeXt,
EfficientNet, DenseNet, RegNet, MobileNetV3, VGG, MaxViT, and more.
ViT / DeiT / Swin variants live under :file:`timm/vit/`.

.. code-block:: console

   $ torchgeo-bench run model=timm/resnet50
   $ torchgeo-bench run model=timm/convnext_base dataset.names=[m-eurosat]
   $ torchgeo-bench run model=timm/vit/vit_base_patch16_224 dataset.image_size=224
   $ torchgeo-bench run model=timm/vit/swin_base_patch4_window7_224 eval.skip_linear=true

ViT-style backbones expect a fixed spatial resolution.  Set
``dataset.image_size=224`` (``bilinear`` by default; switch to
``bicubic`` / ``nearest`` via ``dataset.interpolation``) to resize the
dataset tiles for any model.

timm models rebuild their input convolution for any number of channels —
they work with ``dataset.bands=all`` out of the box (pretrained
3-channel weights are averaged / replicated as needed).

torchgeo foundation models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Configs under :file:`src/torchgeo_bench/conf/model/torchgeo/`.  Most are
RGB-only self-supervised checkpoints from torchgeo's model hub.

.. code-block:: console

   $ # Sentinel-2 RGB SSL
   $ torchgeo-bench run model=torchgeo/resnet50_s2rgb_moco
   $ torchgeo-bench run model=torchgeo/resnet18_s2rgb_seco
   $ torchgeo-bench run model=torchgeo/resnet50_fmow_gassl

   $ # ScaleMAE on fMoW RGB
   $ torchgeo-bench run model=torchgeo/scalemae_large_fmow

   $ # DOFA — band-agnostic (currently configured for Sentinel-2 RGB wavelengths)
   $ torchgeo-bench run model=torchgeo/dofa_base

   $ # Satlas Swin-V2 (NAIP / Sentinel-2 RGB)
   $ torchgeo-bench run model=torchgeo/swinv2b_naip_satlas_mi
   $ torchgeo-bench run model=torchgeo/swinv2b_s2rgb_satlas_mi

   $ # EarthLoc place-recognition descriptor
   $ torchgeo-bench run model=torchgeo/earthloc_s2_resnet50

OlmoEarth (AI2)
^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.OlmoEarthBenchModel`.  Requires the
optional ``olmoearth`` extra:

.. code-block:: console

   $ pip install 'torchgeo-bench[olmoearth]'
   $ torchgeo-bench run model=olmoearth_base
   $ torchgeo-bench run model=olmoearth_large dataset.bands=all

.. note::

   Input normalization is selected globally with ``dataset.normalization``
   (default ``bandspec_zscore``).  Each model receives that strategy through
   :class:`~torchgeo_bench.models.BenchModel`; use ``model_native`` for
   wrappers that declare pretrained input units / statistics, or ``identity``
   when a backbone owns all normalization internally.

SAM 3 vision encoder
^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.SAM3Encoder`.  Requires the optional
``sam3`` extra and a local checkpoint at :file:`checkpoints/sam3/`:

.. code-block:: console

   $ pip install 'torchgeo-bench[sam3]'
   $ torchgeo-bench run model=sam3_encoder dataset.bands=[red,green,blue]

Adding a new model
------------------

1. Implement :class:`~torchgeo_bench.models.BenchModel` in any importable
   module:

   .. code-block:: python

      import torch

      from torchgeo_bench.datasets.base import BandSpec
      from torchgeo_bench.models.interface import BenchModel


      class MyModel(BenchModel):
          def __init__(self, bands: list[BandSpec], pretrained: bool = True):
              super().__init__(bands=bands)
              # self.num_channels == len(bands) is set for you
              self.backbone = create_my_backbone(
                  in_channels=self.num_channels, pretrained=pretrained
              )

          def _forward_patch_features(
              self,
              images: torch.Tensor,
              bboxes: torch.Tensor | None = None,
          ) -> torch.Tensor:
              # `images` has already been normalized via self.normalize_inputs
              return self.backbone(images)  # must return (B, K)

   Notes:

   * ``BenchModel.__init__`` takes a ``bands: list[BandSpec]`` argument; the
     runner builds it from the dataset wrapper and injects it for you.
     Do **not** put ``bands`` in your YAML.
   * The public ``forward_patch_features`` is sealed and applies
     ``normalize_inputs`` (per-channel z-score from each ``BandSpec``'s
     ``mean`` / ``std``) before dispatching to your
     ``_forward_patch_features``.  Override ``normalize_inputs``
     if your backbone expects a different policy (e.g. backbone-internal
     normalization → return ``images`` unchanged).

2. Drop a config at :file:`src/torchgeo_bench/conf/model/<name>.yaml`:

   .. code-block:: yaml

      _target_: my_pkg.MyModel
      pretrained: true
      name: my_model

3. Run it:

   .. code-block:: console

      $ torchgeo-bench run model=<name>

For segmentation models, also pick the
:attr:`eval.segmentation.layers <torchgeo_bench.segmentation_probe.SegmentationProbe>`
that the head will hook into — see :doc:`segmentation-layers` for
verified values per timm backbone family.
