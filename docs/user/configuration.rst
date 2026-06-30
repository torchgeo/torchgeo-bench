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

Config tree
-----------

The packaged config tree is shipped inside the wheel:

.. code-block:: text

   src/torchgeo_bench/conf/
   ├── config.yaml          # primary config (defaults below)
   └── model/
       ├── rcf.yaml
       ├── imagestats.yaml
       ├── sam3_encoder.yaml
       ├── olmoearth_{base,large}.yaml
       ├── timm/
       │   └── ...          # ResNet, ConvNeXt, EfficientNet, ViT/Swin/DeiT, ...
       └── torchgeo/
           └── ...          # SSL backbones, ScaleMAE, DOFA, Satlas, EarthLoc, ...

See :doc:`models` for an operator-facing tour of the available presets.

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
     c_range: [-6, 4, 40]          # log10 sweep start, stop, num samples for linear probe
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

.. seealso::

   :doc:`segmentation-layers` lists the verified ``layers`` values for
   every supported timm backbone family, with spatial sizes and notes on
   stages that share resolution (common in EfficientNet / MobileNet).

Refer to :doc:`/api/eval` for the runtime functions that consume each
sub-block.

Hyperparameter search (segmentation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian hyperparameter optimisation over learning rate and weight decay is
available for the segmentation probe.  It requires Optuna:

.. code-block:: console

   $ pip install torchgeo-bench[hpo]

Enable it via ``eval.segmentation.hparam_search=true``.  The full set of
tunable options with their defaults:

.. code-block:: yaml

   eval:
     segmentation:
       hparam_search: false     # enable Bayesian HPO (requires torchgeo-bench[hpo])
       n_trials: 10             # Optuna TPE trials
       hpo_epochs: 5            # training epochs per trial
       lr_min: 1e-5             # LR search lower bound (log-uniform)
       lr_max: 1e-2             # LR search upper bound
       wd_min: 1e-6             # weight-decay search lower bound (log-uniform)
       wd_max: 1e-1             # weight-decay search upper bound

Example CLI invocation:

.. code-block:: console

   $ torchgeo-bench run \
       model=timm/resnet50 \
       eval.segmentation.hparam_search=true \
       eval.segmentation.n_trials=20

After HPO the head is retrained from scratch on the merged train+val split
using the best found parameters for the full ``epochs``.  The results CSV
stores the winning values in the ``best_lr`` and ``best_weight_decay``
columns; both are ``null`` when HPO is disabled.

``model`` block
---------------

The default ``model: rcf`` selects Random Convolutional Features.  Browse
:file:`src/torchgeo_bench/conf/model/` for the full list of presets.  Each
preset can ship its own ``eval`` overrides (for example, segmentation
backbones often pin ``head_type: fpn`` and bump the learning rate); these
are merged on top of the global ``eval`` block per dataset.
