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
``--model``/``-m`` selector for the ``run`` subcommand.  A preset's
``_target_`` field resolves to a class re-exported from
:mod:`torchgeo_bench.models`.

Random Convolutional Features (RCF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.RCFBench`. Gaussian or empirical random
features in the spirit of MOSAIKS.

.. code-block:: console

   $ torchgeo-bench run --model rcf

Fields with no dedicated CLI flag (like RCF's ``mode``/``features``) are
set by copying :file:`conf/model/rcf.yaml` to a new preset name and editing
it -- there is no ``key=value`` override for arbitrary model kwargs.

Image statistics baseline
^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.ImageStatsBench`. A trivial baseline that
returns per-channel mean / std as the feature vector.

.. code-block:: console

   $ torchgeo-bench run --model imagestats

timm — ImageNet-pretrained CNNs and ViTs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.TimmPatchBenchModel`.  Configs under
:file:`src/torchgeo_bench/conf/model/timm/` cover ResNet, ConvNeXt,
EfficientNet, DenseNet, RegNet, MobileNetV3, VGG, MaxViT, and more.
ViT / DeiT / Swin variants live under :file:`timm/vit/`.

.. code-block:: console

   $ torchgeo-bench run --model timm/resnet50
   $ torchgeo-bench run --model timm/convnext_base --datasets m-eurosat
   $ torchgeo-bench run --model timm/vit/vit_base_patch16_224 --image-size 224
   $ torchgeo-bench run --model timm/vit/swin_base_patch4_window7_224 --skip-linear

ViT-style backbones expect a fixed spatial resolution.  Set
``--image-size 224`` (``bilinear`` by default; switch to
``bicubic`` / ``nearest`` via ``--interpolation``) to resize the
dataset tiles for any model.

timm models rebuild their input convolution for any number of channels —
they work with ``--bands all`` out of the box (pretrained
3-channel weights are averaged / replicated as needed).

.. warning::

   ``timm/efficientnet_b1``'s ``linear`` probe is unreliable across most
   classification datasets (``knn5`` is fine). The C sweep in
   ``evaluate_logistic`` doesn't standardize features before fitting, and
   b1's feature geometry makes the fit unstable across the whole
   ``c_range`` grid rather than diverging outright — so it lands on a poor
   but finite ``best_c`` instead of raising
   ``LinearProbeDivergedError``. Treat
   ``efficientnet_b1`` linear numbers as noise until the probe gains
   feature standardization; other EfficientNet variants (b0, b2-b4) are
   unaffected.

torchgeo foundation models
^^^^^^^^^^^^^^^^^^^^^^^^^^

Configs under :file:`src/torchgeo_bench/conf/model/torchgeo/`.  Most are
RGB-only self-supervised checkpoints from torchgeo's model hub.

.. code-block:: console

   $ # Sentinel-2 RGB SSL
   $ torchgeo-bench run --model torchgeo/resnet50_s2rgb_moco
   $ torchgeo-bench run --model torchgeo/resnet18_s2rgb_seco
   $ torchgeo-bench run --model torchgeo/resnet50_fmow_gassl

   $ # ScaleMAE on fMoW RGB
   $ torchgeo-bench run --model torchgeo/scalemae_large_fmow

   $ # DOFA — band-agnostic (currently configured for Sentinel-2 RGB wavelengths)
   $ torchgeo-bench run --model torchgeo/dofa_base

   $ # Satlas Swin-V2 (NAIP / Sentinel-2 RGB)
   $ torchgeo-bench run --model torchgeo/swinv2b_naip_satlas_mi
   $ torchgeo-bench run --model torchgeo/swinv2b_s2rgb_satlas_mi

   $ # EarthLoc place-recognition descriptor
   $ torchgeo-bench run --model torchgeo/earthloc_s2_resnet50

OlmoEarth (AI2)
^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.OlmoEarthBenchModel`.  Requires the
optional ``olmoearth`` extra:

.. code-block:: console

   $ pip install 'torchgeo-bench[olmoearth]'

   $ # OlmoEarth v1 (Nano / Tiny / Base / Large)
   $ torchgeo-bench run --model olmoearth_nano
   $ torchgeo-bench run --model olmoearth_base
   $ torchgeo-bench run --model olmoearth_large --bands all

   $ # OlmoEarth v1.1 (Nano / Tiny / Base)
   $ torchgeo-bench run --model olmoearth_v1_1_nano
   $ torchgeo-bench run --model olmoearth_v1_1_tiny
   $ torchgeo-bench run --model olmoearth_v1_1_base

   $ # OlmoEarth v1.2 (Nano / Tiny / Small / Base)
   $ torchgeo-bench run --model olmoearth_v1_2_nano
   $ torchgeo-bench run --model olmoearth_v1_2_small
   $ torchgeo-bench run --model olmoearth_v1_2_base

OlmoEarth v1.1 uses a **linear patch embedding** (vs. convolutional in v1),
a single bandset per modality, and updated masking/loss functions, yielding a
≈ 3× reduction in MACs with comparable accuracy.  OlmoEarth v1.2 adds **RoPE
3D position encoding** and a new **Small** size (384-d) between Tiny and Base.
The ``version`` parameter selects the weight family:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Config
     - Version
     - Size
     - Notes
   * - ``olmoearth_nano``
     - v1
     - Nano
     - multi-bandset, conv patch embed
   * - ``olmoearth_tiny``
     - v1
     - Tiny
     -
   * - ``olmoearth_base``
     - v1
     - Base
     -
   * - ``olmoearth_large``
     - v1
     - Large
     -
   * - ``olmoearth_v1_1_nano``
     - v1.1
     - Nano
     - single-bandset, linear patch embed
   * - ``olmoearth_v1_1_tiny``
     - v1.1
     - Tiny
     -
   * - ``olmoearth_v1_1_base``
     - v1.1
     - Base
     -
   * - ``olmoearth_v1_2_nano``
     - v1.2
     - Nano
     - RoPE position encoding
   * - ``olmoearth_v1_2_tiny``
     - v1.2
     - Tiny
     -
   * - ``olmoearth_v1_2_small``
     - v1.2
     - Small
     - new size (384-d, ≈ 35.6M params)
   * - ``olmoearth_v1_2_base``
     - v1.2
     - Base
     -

.. note::

   Input normalization is selected globally with ``--normalization``
   (default ``bandspec_zscore``).  Each model receives that strategy through
   :class:`~torchgeo_bench.models.BenchModel`; use ``model_native`` for
   wrappers that declare pretrained input units / statistics, or ``identity``
   when a backbone owns all normalization internally.

   GeoBench delivers Landsat imagery (e.g. ``m-forestnet``) as uint8
   [0, 255], a scale OlmoEarth's pretrained Landsat statistics (fit on real
   DN) can't match.  OlmoEarth therefore selects normalization per sensor
   (``norm_from_pretrained="auto"``, the default): Landsat is normalized with
   dataset-specific ``BandSpec`` stats while Sentinel-2 / SAR use the
   pretrained normalizer.  ``norm_from_pretrained`` has no dedicated CLI
   flag; to force one path for all sensors, copy the relevant OlmoEarth
   preset YAML to a new name and set ``norm_from_pretrained: true`` (or
   ``false``) in the copy.

.. note::

   **Per-model input resolution.**  A model preset may set ``image_size`` to
   override the global default (``--image-size``, ``224``).  OlmoEarth is
   resolution-flexible, so its presets set ``image_size: null`` to evaluate at
   each dataset's **native** resolution rather than upsampling to 224×224
   (matching the reference OlmoEarth evals).  Models that omit the field
   inherit the ``--image-size`` value.  To force a specific size for a run,
   pass ``--image-size <int>`` on the command line.  The effective size is
   recorded in the results CSV and in the resume cache key.

SAM 3 vision encoder
^^^^^^^^^^^^^^^^^^^^

:class:`~torchgeo_bench.models.SAM3Encoder`.  Requires the optional
``sam3`` extra and a local checkpoint at :file:`checkpoints/sam3/`:

.. code-block:: console

   $ pip install 'torchgeo-bench[sam3]'
   $ torchgeo-bench run --model sam3_encoder --bands red,green,blue

Adding a new model
------------------

There are two contribution pathways.  **Stage 1** lets you benchmark your
model locally and report results in a paper without opening a PR.  **Stage
2** covers the full code contribution: exporting the class, writing tests,
hosting weights, and submitting a PR.

.. seealso::

   :doc:`eval_own_model`
      Stage 1 — evaluate your model locally and report results.

   :doc:`contribute_model`
      Stage 2 — contribute the model as a PR to the shared benchmark.

.. note::

   Two key patterns apply regardless of stage:

   * **Do not put** ``bands`` **in the model YAML.**  The runner reads the
     current dataset's :class:`~torchgeo_bench.datasets.BandSpec` list
     and injects it into the constructor automatically.  Adding ``bands`` to
     the YAML causes a ``TypeError`` (duplicate keyword argument).
   * **Pass** ``normalization="identity"`` **to** ``super().__init__`` **when
     your backbone handles normalization internally** (e.g. OlmoEarth, Clay,
     any model whose ``forward()`` runs its own per-channel standardization).
     The sealed ``forward_patch_features`` will then pass raw sensor values
     straight to your ``_forward_patch_features`` without applying any
     additional z-score.

For segmentation models, also pick the
:attr:`eval.segmentation.layers <torchgeo_bench.segmentation_probe.SegmentationProbe>`
that the head will hook into — see :doc:`segmentation-layers` for
verified values per timm backbone family.
