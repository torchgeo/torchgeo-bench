Configuration
=============

``torchgeo-bench`` is configured with a plain ``argparse`` CLI. Settings are
composed in this order (each stage overrides the previous one):

#. built-in Python defaults (``torchgeo_bench.settings``),
#. an optional ``--config PATH`` YAML file of uncommon settings,
#. the selected model preset YAML (``--model``, from
   :file:`src/torchgeo_bench/conf/model/`), and
#. explicit CLI flags, which always win.

There is no dotted-path ``key=value`` override syntax. Common settings have
their own flags (see ``torchgeo-bench run --help``); anything else goes in
the ``--config`` YAML:

.. code-block:: console

   $ torchgeo-bench run \
       --model timm/resnet50 \
       --datasets m-eurosat \
       --bootstrap 100 \
       --skip-linear \
       --device cuda:1

Use ``--print-config`` to print the fully merged settings (in the same YAML
shape ``--config`` expects) and exit without running anything.

Model preset tree
------------------

The packaged preset tree is shipped inside the wheel:

.. code-block:: text

   src/torchgeo_bench/conf/
   └── model/
       ├── rcf.yaml
       ├── imagestats.yaml
       ├── mind.yaml
       ├── sincos.yaml
       ├── sam3_encoder.yaml
       ├── olmoearth_{base,large}.yaml
       ├── timm/
       │   └── ...          # ResNet, ConvNeXt, EfficientNet, ViT/Swin/DeiT, ...
       └── torchgeo/
           └── ...          # SSL backbones, ScaleMAE, DOFA, Satlas, EarthLoc, ...

Each preset is a YAML file with a ``_target_`` (a dotted import path) plus
constructor kwargs; ``--model-help <name>`` prints one without running
anything. See :doc:`models` for an operator-facing tour of the available
presets.

Common flags (``torchgeo-bench run``)
--------------------------------------

============================  ==================================================
Flag                          Meaning
============================  ==================================================
``-m``, ``--model``           Model preset, e.g. ``timm/resnet50`` (default ``rcf``).
``-d``, ``--datasets``        Comma-separated dataset names, or ``all``.
``--device``                  PyTorch device string (e.g. ``cuda:0``, ``cpu``; default: auto).
``-o``, ``--output``          Results CSV path (default: ``results/models/<model name>.csv``).
``--resume``                  Skip already-computed ``(dataset, method, model, config)`` combos.
``--seed``                    Global RNG seed (numpy + torch).
``-v``, ``--verbose``         Toggle INFO-level progress logging.
============================  ==================================================

``torchgeo-bench run`` (and the ``profile``/``intrinsic-dim`` aliases) also
accept ``--partition``, ``--bands``, ``--batch-size``, ``--num-workers``,
``--image-size``, ``--time-steps``, ``--interpolation``, ``--normalization``,
``--skip-linear``, ``--bootstrap``, ``--merge-val``/``--no-merge-val``,
``--knn-device``, ``--seg-head``, ``--seg-epochs``, ``--seg-lr``,
``--seg-scheduler``, ``--seg-batch-size``, ``--seg-cache``/``--no-seg-cache``,
``--seg-cache-dtype``, ``--use-cls-token``/``--no-use-cls-token``,
``--model-input-normalization``, and ``--model-name``. Run
``torchgeo-bench run --help`` for the authoritative list with defaults.

See :doc:`datasets` for the full list of available dataset names and band
selection semantics.

Uncommon settings (``--config`` YAML)
--------------------------------------

Settings without their own flag are set via a YAML file passed to
``--config``, merged under the built-in defaults but under the model preset
and any explicit flags. Its shape mirrors ``--print-config`` output, for
example:

.. code-block:: yaml

   eval:
     c_range: [-6, 4, 40]          # log10 sweep start, stop, num samples for linear probe
     calibration:
       n_bins_knn: null            # null = knn_k + 1
       n_bins_linear: 15
       temp_scale: false           # requires --no-merge-val (held-out validation logits)
     segmentation:
       criterion:
         _target_: torch.nn.CrossEntropyLoss
         ignore_index: 255
       save_viz: true
       viz_dir: viz
       n_viz_samples: 8

The additive intrinsic-dimension and compute-profile passes are run via the
dedicated ``torchgeo-bench intrinsic-dim`` / ``torchgeo-bench profile``
subcommands (which set ``eval.intrinsic_dim.enabled`` /
``eval.profile.enabled`` for you); their own sub-settings (estimators,
splits, warmup/measure counts, ...) are set the same way, through
``--config``.

.. seealso::

   :doc:`segmentation-layers` lists the verified ``--seg-head``-compatible
   layer names for every supported timm backbone family, with spatial sizes
   and notes on stages that share resolution (common in EfficientNet /
   MobileNet).

Refer to :doc:`/api/eval` for the runtime functions that consume each
setting.

CoordBench
----------

The coordinate-only track is a dedicated subcommand,
``torchgeo-bench coord``, with its own flags (``--model``, ``--names``,
``--methods``, ``--split``, ``--folds``, ``--cell-deg``, ``--knn-k``,
``--knn-device``, ``--output``, ``--resume``, ``--seed``, ``--config``,
``--print-config``):

.. code-block:: console

   $ torchgeo-bench coord --model sincos --names all --methods knn,linear --split random

See :doc:`coordbench` for benchmark families, included encoders, split
semantics, and a custom-encoder example.
