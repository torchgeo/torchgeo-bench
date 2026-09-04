# R06 - Choose normalization ownership and explicit input units

Status: proposed; no option has been accepted. This file changes no runtime behavior.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/312

Replace guessed units and competing wrapper settings with explicit metadata and auditable preprocessing.

## Decision

Should one preprocessing operation combine dataset/model metadata, or should datasets and models own separate conversion stages?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | One resolved operation (recommended) | Datasets expose decoded values and metadata; apply the requested policy exactly once. |
| B | Dataset conversion then model preprocessing | Datasets emit canonical per-modality units; record and validate both stages. |

Vote in the linked issue. Comment `Vote: A` or `Vote: B` with a short rationale. If neither fits, propose a concrete amendment. Recommendations are proposals, not recorded votes or maintainer approval. Maintainers will summarize the outcome in the issue. Reactions indicate interest, not a choice between options.

## Proposed contract

Declare units, scale/offset, nodata, band identity, wavelength where applicable, and statistics provenance per band/modality. Dtype and observed maxima cannot establish physical units. RGB divided by 255 is scaled RGB, not calibrated reflectance. Optical and SAR inputs require separate metadata.

For option A, datasets emit decoded values with metadata. Resolve one preprocessing operation from that metadata, model requirements, and requested policy. Apply it once for both classification and segmentation. Do not normalize in datasets and undo it in model wrappers.

| Proposed policy | Operation |
| --- | --- |
| `dataset` | Standardize using declared per-band statistics in matching units. |
| `model` | Apply documented checkpoint preprocessing after supported explicit unit conversion. |
| `minmax` | Scale using declared lower/upper bounds; clipping is separately specified. |
| `none` | Preserve decoded values without numerical normalization. |

Retain dataset standardization as the migration default. Changing the scientific default requires separate evidence and a decision. For option B, canonical units are defined per modality; a universal [0, 1] range is not a sufficient contract for all sensors and processing levels.

Band reordering also reorders statistics. Record their split, revision, units, and source. Fit new statistics on training data only. Keep upstream statistics with uncertain provenance explicitly marked as legacy until audited; do not relabel them as verified training statistics.

Identify nodata before normalization, exclude invalid pixels from fitted statistics, preserve ignored target labels, and declare the fill passed to the backbone. Specify zero-variance/zero-span handling, clipping, resize interpolation, and missing bands rather than silently choosing epsilon or padding behavior.

Inseparable vendor normalizers support `model` explicitly and reject incompatible policies. A valid model policy may be unit conversion alone when that matches documented training inputs; not every checkpoint requires mean/std standardization. Record the actual conversions, statistics, and vendor-preprocessing version. Feature L2 normalization is separate.

`minmax_zscore` reduces algebraically to z-score for consistent nondegenerate statistics. Check legacy clamps and rounding before retiring it. Version intentional differences and do not rewrite historical result labels.

## Acceptance criteria for implementation

- [ ] Reference cases cover raw/scaled S2, uint8 imagery, SAR/optical mixtures, and reordered bands.
- [ ] Tests prove one normalization pass, no validation/test fitting, and accurate tensor metadata.
- [ ] Unsupported conversions, absent statistics, nodata, and degenerate bands have documented behavior.
- [ ] Preprocessed tensors and predictions match the oracle unless a separately documented protocol correction explains the difference.

## Review boundary

Units, normalization ownership/provenance, and representative adapters. New benchmark defaults and broad wrapper rewrites remain separate.

The draft implements `ImageNormalizer` with explicit `InputBand`, `ModelBand`, and `Statistics` records. It accepts BCHW and BTCHW tensors, fills invalid pixels after normalization, rejects degenerate statistics, and makes clipping opt-in.

Supported conversions are declared `s2_dn` to/from reflectance and uint8 to/from scaled RGB. Decoders must apply file calibration offsets before this operation; these names do not establish an arbitrary sensor product's calibration. SAR log/linear and unknown cross-unit conversions fail explicitly.

A CPU test compares a real ResNet-18 encoder preceded by this normalizer with the current wrapper's embeddings. The new path disables the wrapper's legacy normalizer. Other tensor tests cover mixed optical/SAR channels, channel order, min-max clipping, nodata, and metadata errors.

This is an initial implementation for review. Existing benchmark wrappers keep their current defaults until their dataset/checkpoint units and statistics have been verified. No existing statistics are relabeled as verified training statistics, and no historical result labels are rewritten.

## Existing work and references

Continue [#259](https://github.com/torchgeo/torchgeo-bench/issues/259) and [#260](https://github.com/torchgeo/torchgeo-bench/issues/260); consider [#256](https://github.com/torchgeo/torchgeo-bench/issues/256) and DEO [#243](https://github.com/torchgeo/torchgeo-bench/pull/243). Opening this issue does not decide a new default.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Unit heuristics](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/models/_input_units.py#L25)
- [Current policies](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/models/_normalization.py#L1)
- [timm override](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/models/timm.py#L217)
