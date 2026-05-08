Configuration
=============

``torchgeo-bench`` is driven by `Hydra <https://hydra.cc>`_.  The primary
config lives at :file:`src/torchgeo_bench/conf/config.yaml` and is composed
with a model preset selected from :file:`src/torchgeo_bench/conf/model/`.

Every value in the config can be overridden on the command line using
Hydra's dotted-path syntax:

.. code-block:: console

   $ torchgeo-bench run \
       model=timm/resnet50 \
       dataset.names=[m-eurosat] \
       eval.bootstrap=100 \
       eval.skip_linear=true \
       device=cuda:1

Top-level options
-----------------

============================  ==================================================
Key                           Meaning
============================  ==================================================
``seed``                      Global RNG seed (numpy + torch).
``device``                    PyTorch device string (e.g. ``cuda:0``, ``cpu``).
``output``                    Path to the appended results CSV.
``verbose``                   Toggle progress logging.
``resume``                    Skip already-computed ``(dataset, method, model, config)`` combos.
============================  ==================================================

``dataset`` block
-----------------

.. code-block:: yaml

   dataset:
     names: all                    # or a list, e.g. [m-eurosat, m-pv4ger]
     partition: default            # alternative GeoBench V1 partitions when supported
     batch_size: 64
     bands: rgb                    # rgb | all | [red, green, blue, nir]
     image_size: 224               # null disables resizing
     interpolation: bilinear       # bilinear | bicubic | nearest

See :doc:`datasets` for the full list of available dataset names and band
selection semantics.

``eval`` block
--------------

.. code-block:: yaml

   eval:
     bootstrap: 200                # bootstrap resamples for KNN/linear CIs
     c_range: [-7, 2, 20]          # log10 sweep start, stop, num samples for linear probe
     merge_val: true               # merge train+val before training the final logistic head
     skip_linear: false            # skip the (slower) linear probe entirely

     intrinsic_dim:                # optional ID metrics on extracted embeddings
       enabled: false
       estimators: [TwoNN, MLE, lPCA]
       splits: [train]
       max_samples: 10000
       device: null                # null = auto (cuda if available)

     segmentation:                 # used only for segmentation datasets
       head_type: fpn              # linear | fpn | dpt | conv-block
       layers: []                  # backbone layers to extract; [] = use defaults
       lr: 1e-3
       epochs: 10
       criterion:
         _target_: torch.nn.CrossEntropyLoss
         ignore_index: 255
       lr_scheduler: cosine
       cache_features: true
       cache_dtype: float16
       save_viz: false
       viz_dir: viz
       n_viz_samples: 8

Refer to :doc:`/api/eval` for the runtime functions that consume each
sub-block.

``model`` block
---------------

The default ``model: rcf`` selects Random Convolutional Features.  Browse
:file:`src/torchgeo_bench/conf/model/` for the full list of presets.  Each
preset can ship its own ``eval`` overrides (for example, segmentation
backbones often pin ``head_type: fpn`` and bump the learning rate); these
are merged on top of the global ``eval`` block per dataset.
