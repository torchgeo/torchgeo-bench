"""Shared scaffolding for torchgeo pretrained-weights wrappers.

This module holds the genuinely stable mechanisms every ``torchgeo_*``
family module builds on: factory/weights resolution, first-conv channel
adaptation, auto-resize, pretraining-normalization extraction, and the
input-unit plausibility warning.  Family modules (ResNet/Swin, ScaleMAE,
DOFA/EarthLoc, DEO, CROMA/Panopticon) import from here rather than
duplicating this logic.

Caveats
-------

The pretrained weights' ``Normalize`` transform was calibrated for a
specific input scale (e.g. Sentinel-2 DN / 10000, NAIP uint8 / 255).
Pairing one of these wrappers with a dataset whose raw values are in a
different scale will silently misnormalize.  Each wrapper sets
:attr:`weights_input_unit` documenting the expected scale, and
:func:`_warn_unit_mismatch` emits a warning when the band statistics
look incompatible.  See GitHub issue
`#16 <https://github.com/torchgeo/torchgeo-bench/issues/16>`_ for the
follow-up on stronger guards.
"""

import logging
import warnings
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeo.models as tgm
from torchvision.transforms import Normalize as NormalizeV1
from torchvision.transforms.v2 import Normalize as NormalizeV2

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import InputUnit, convert_unit, detect_input_unit
from ._normalization import NormalizationStrategy
from .interface import BenchModel

logger = logging.getLogger(__name__)

#: Polarization/sensor tags treated as SAR (non-optical) by the DOFA and
#: Panopticon wrappers' wavelength/channel-id fallbacks.
_SAR_SENSORS = frozenset({"s1", "sar"})


def _resolve_torchgeo_factory(factory_name: str):
    """Return the model-factory function from ``torchgeo.models``."""
    fn = getattr(tgm, factory_name, None)
    if fn is None:
        raise ValueError(f"torchgeo.models has no factory function '{factory_name}'")
    return fn


def _resolve_torchgeo_weights(weights_class_name: str, weights_member: str):
    """Return the concrete weights enum member."""
    cls = getattr(tgm, weights_class_name, None)
    if cls is None:
        raise ValueError(f"torchgeo.models has no weights class '{weights_class_name}'")
    member = getattr(cls, weights_member, None)
    if member is None:
        raise ValueError(f"{weights_class_name} has no member '{weights_member}'")
    return member


def _adapt_first_conv(model: nn.Module, attr_path: str, in_chans: int) -> None:
    """Adapt ``model.<attr_path>`` (a ``Conv2d``) to ``in_chans`` input channels.

    Reuses :func:`timm.models._manipulate.adapt_input_conv` for the shapes it
    supports: an RGB-pretrained source stem (``conv.in_channels == 3``,
    arbitrary target) or an arbitrary source stem averaged down to a single
    target channel (``in_chans == 1``).  Any other source shape (e.g. 13ch
    MoCo-MSI -> 18ch S1+S2) is unsupported by timm's helper, so it is routed
    directly to the documented fallback instead: average the pretrained
    weight to one channel and replicate it with a ``3 / in_chans`` scale to
    preserve activation magnitude.
    """
    from timm.models._manipulate import adapt_input_conv

    parts = attr_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    conv = getattr(parent, parts[-1])
    if conv.in_channels == in_chans:
        return

    timm_adaptation_supported = conv.in_channels == 3 or in_chans == 1
    new_weight = None
    if timm_adaptation_supported:
        new_weight = adapt_input_conv(in_chans, conv.weight.data)
    if new_weight is None or new_weight.shape[1] != in_chans:
        # Fallback: average pretrained weight to a single channel then
        # replicate, scaling by the original-to-target channel ratio so
        # the post-conv activation magnitude is preserved.
        avg = conv.weight.data.float().mean(dim=1, keepdim=True)
        new_weight = avg.expand(-1, in_chans, -1, -1).contiguous()
        new_weight = new_weight * (conv.in_channels / float(in_chans))
        new_weight = new_weight.to(conv.weight.dtype)

    new_conv = nn.Conv2d(
        in_channels=in_chans,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
    )
    new_conv.weight.data.copy_(new_weight)
    if conv.bias is not None:
        new_conv.bias.data.copy_(conv.bias.data)
    setattr(parent, parts[-1], new_conv)


def _auto_resize(images: torch.Tensor, target_size: int) -> torch.Tensor:
    h, w = images.shape[-2], images.shape[-1]
    if h != target_size or w != target_size:
        images = F.interpolate(
            images,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
    return images


def _extract_normalize_transforms(weights) -> nn.Sequential | None:
    """Extract only the ``Normalize`` layers from a torchgeo weights transform."""
    if not hasattr(weights, "transforms") or weights.transforms is None:
        return None
    transform = weights.transforms
    if callable(transform) and not isinstance(transform, nn.Module):
        transform = transform()
    if isinstance(transform, nn.Identity):
        return None
    try:
        iterator = iter(transform)
    except TypeError:
        return None
    norms = [t for t in iterator if isinstance(t, (NormalizeV1, NormalizeV2))]
    if not norms:
        return None
    return nn.Sequential(*norms)


# Magnitude buckets for `weights_input_unit` plausibility checks.  Keys are
# rough expected per-band mean ranges in raw units.
_UNIT_EXPECTED_MEAN: dict[str, tuple[float, float]] = {
    "uint8_div255": (0.0, 255.0),
    "reflectance_0_1": (0.0, 2.0),
    "s2_dn_div10000": (0.0, 10000.0),
}

_UNIT_EXPECTED_SOURCE: dict[str, InputUnit] = {
    "uint8_div255": InputUnit.UINT8,
    "reflectance_0_1": InputUnit.REFLECTANCE_0_1,
    "s2_dn_div10000": InputUnit.S2_DN,
}


def _warn_unit_mismatch(
    cls_name: str,
    weights_input_unit: str | None,
    bands: list[BandSpec],
    check: str,
) -> None:
    """Emit a warning if the per-band ``mean`` magnitude looks incompatible.

    Args:
        cls_name: Wrapper class name, used in the warning message.
        weights_input_unit: Expected input scale tag (key into
            :data:`_UNIT_EXPECTED_MEAN`).  ``None`` skips the check.
        bands: The dataset's :class:`BandSpec` list.
        check: ``"warn"`` (default) emits a UserWarning; ``"error"`` raises;
            ``"ignore"`` is silent.
    """
    if check == "ignore" or weights_input_unit is None:
        return
    expected_unit = _UNIT_EXPECTED_SOURCE.get(weights_input_unit)
    detected_unit = detect_input_unit(bands)
    if expected_unit is not None and detected_unit != expected_unit:
        msg = (
            f"{cls_name}: pretrained weights expect {weights_input_unit!r} inputs "
            f"({expected_unit.value}), but selected bands look like {detected_unit.value}: "
            f"{[(b.name, b.mean, b.max) for b in bands[:5]]}"
            f"{'...' if len(bands) > 5 else ''}. Embeddings may be poorly scaled."
        )
        if check == "error":
            raise RuntimeError(msg)
        warnings.warn(msg, UserWarning, stacklevel=3)
        return
    expected = _UNIT_EXPECTED_MEAN.get(weights_input_unit)
    if expected is None:
        return
    lo, hi = expected
    bad = [b for b in bands if not (lo <= b.mean <= hi * 1.5)]
    if not bad:
        return
    msg = (
        f"{cls_name}: pretrained weights expect inputs in unit "
        f"{weights_input_unit!r} (per-band mean ~ [{lo}, {hi}]), but the "
        f"selected dataset has bands with mean outside that range: "
        f"{[(b.name, b.mean) for b in bad[:5]]}{'...' if len(bad) > 5 else ''}. "
        "Embeddings may be poorly scaled."
    )
    if check == "error":
        raise RuntimeError(msg)
    warnings.warn(msg, UserWarning, stacklevel=3)


class _TorchGeoBackboneBench(BenchModel):
    """Shared scaffolding for torchgeo pretrained-weights wrappers.

    Subclasses set :attr:`weights_input_unit` and implement their feature
    extraction from the loaded backbone.
    """

    weights_input_unit: str | None = None

    # `model_native`'s normalizer here is the loaded weights' own Normalize
    # transform, which isn't known until partway through this __init__ (after
    # the backbone/weights are resolved).  Finalize it explicitly once that's
    # ready, instead of letting the base class attempt (and, for wrappers
    # with no pretrain_mean, immediately fail to build) one too early.
    installs_own_model_native_normalizer = True

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str,
        weights_class: str,
        weights_member: str,
        auto_resize: bool,
        target_size: int | None,
        normalization_input_unit: str | None = None,
        skip_weight_normalize: int = 0,
        input_unit_check: str = "warn",
        factory_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # The pretraining scale is already named by weights_input_unit, so
        # derive expected_input_unit from it rather than making every wrapper
        # restate it — model_native fails without it.
        if type(self).expected_input_unit is None and self.weights_input_unit:
            derived = _UNIT_EXPECTED_SOURCE.get(self.weights_input_unit)
            if derived is not None:
                type(self).expected_input_unit = derived
        super().__init__(bands=bands, **kwargs)
        if normalization_input_unit is not None:
            if normalization_input_unit not in _UNIT_EXPECTED_SOURCE:
                raise ValueError(
                    "normalization_input_unit must be one of "
                    f"{tuple(_UNIT_EXPECTED_SOURCE)}, got {normalization_input_unit!r}."
                )
            self.normalization_input_unit = normalization_input_unit
        else:
            self.normalization_input_unit = self.weights_input_unit
        if (
            isinstance(skip_weight_normalize, bool)
            or not isinstance(skip_weight_normalize, int)
            or skip_weight_normalize < 0
        ):
            raise ValueError(
                "skip_weight_normalize must be a non-negative integer, "
                f"got {skip_weight_normalize!r}."
            )
        weights = _resolve_torchgeo_weights(weights_class, weights_member)
        self.weights = weights
        model_factory = _resolve_torchgeo_factory(factory)
        self.backbone = model_factory(weights=weights, **(factory_kwargs or {}))
        self.auto_resize = auto_resize
        self.target_size = target_size
        weights_normalize = _extract_normalize_transforms(weights)
        if weights_normalize is not None:
            if skip_weight_normalize > len(weights_normalize):
                raise ValueError(
                    f"skip_weight_normalize={skip_weight_normalize} exceeds the "
                    f"{len(weights_normalize)} Normalize layers supplied by {weights_member}."
                )
            self._weights_normalize = nn.Sequential(
                *list(weights_normalize)[skip_weight_normalize:]
            )
        else:
            self._weights_normalize = None
        if input_unit_check not in ("warn", "ignore", "error"):
            raise ValueError(
                f"input_unit_check must be one of warn|ignore|error, got {input_unit_check!r}."
            )
        native = self.normalization is NormalizationStrategy.MODEL_NATIVE
        if native and normalization_input_unit is None:
            _warn_unit_mismatch(
                type(self).__name__, self.weights_input_unit, bands, input_unit_check
            )

        # Pre-compute the unit conversion needed to bring dataset inputs into
        # the scale the weights' Normalize was calibrated for.  No-op when the
        # wrapper doesn't declare a unit, or the dataset already delivers the
        # expected scale.  Without this, e.g., resnet50_s2rgb_moco × so2sat
        # collapses to chance because the Normalize ``/10000`` is applied to
        # already-reflectance ([0, 2.8]) values, producing near-zero inputs.
        self._weights_target_unit: InputUnit | None = _UNIT_EXPECTED_SOURCE.get(
            self.normalization_input_unit or ""
        )
        self._dataset_input_unit = (
            detect_input_unit(self.bands)
            if native and self._weights_target_unit is not None
            else None
        )
        # Finalize now that the weights (and therefore self._weights_normalize)
        # are known: subclasses whose normalize_inputs always fully implements
        # model_native regardless of the weights' own transform (e.g. CROMA)
        # override _model_native_weights_normalizer() to never return None.
        self.finalize_model_native_normalizer(self._model_native_weights_normalizer())

    def _model_native_weights_normalizer(self) -> nn.Sequential | None:
        """Return this instance's ``model_native`` normalizer, or ``None``.

        Default: the loaded weights' own ``Normalize`` transform (already
        sliced by ``skip_weight_normalize``), or ``None`` if the weights ship
        none.  ``None`` means ``finalize_model_native_normalizer()`` falls
        back to the generic ``pretrain_mean``/``pretrain_std`` builder --
        which most torchgeo wrappers don't set, so it raises immediately.
        Subclasses that always supply their own complete ``model_native``
        pipeline regardless of the weights' transform (e.g. CROMA's
        per-band clip+rescale) should override this to never return
        ``None``.
        """
        return self._weights_normalize

    def _tiled_normalize(self, in_chans: int) -> nn.Sequential | None:
        """Build the pretrained normalization chain for ``in_chans`` channels.

        Matches ``adapt_input_conv``'s tiling pattern: for ``in_chans=7``
        with 3-channel pretrain stats ``[r, g, b]``, the result is
        ``[r, g, b, r, g, b, r]``.  This keeps the input conv (which was
        also tiled) and the normalize statistically consistent — both
        layers "see" each input channel as belonging to the corresponding
        RGB slot of the pretrained model. Every normalization stage is
        preserved; this matters for pipelines such as ScaleMAE's
        ``/255 -> ImageNet z-score`` chain.

        Cached on ``self`` so we don't rebuild per batch.
        """
        cache_key = f"_tiled_norm_{in_chans}"
        cached = getattr(self, cache_key, None)
        if cached is not None:
            return cached
        if self._weights_normalize is None:
            return None
        from torchvision.transforms import Normalize as _N1
        from torchvision.transforms.v2 import Normalize as _N2

        layers: list[nn.Module] = []
        for inner in self._weights_normalize:
            if not isinstance(inner, (_N1, _N2)):
                continue
            mean = inner.mean
            std = inner.std
            if isinstance(mean, torch.Tensor):
                mean = mean.tolist()
            if isinstance(std, torch.Tensor):
                std = std.tolist()
            mean = list(mean)
            std = list(std)
            source_channels = len(mean)
            if source_channels == 0:
                return None
            if source_channels not in (1, in_chans):
                mean = [mean[i % source_channels] for i in range(in_chans)]
                std = [std[i % source_channels] for i in range(in_chans)]
            layers.append(
                type(inner)(
                    mean=mean,
                    std=std,
                    inplace=getattr(inner, "inplace", False),
                )
            )
        if not layers:
            return None
        tiled = nn.Sequential(*layers)
        # Cache on the same device the next forward will use; Normalize is
        # parameter-less so no .to() needed for tensors-on-Tensor input.
        object.__setattr__(self, cache_key, tiled)
        return tiled

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Use the weights-bound ``Normalize`` transform if present; else the parent strategy.

        Pretrained weights ship a 3-channel RGB ``Normalize`` calibrated
        for the pretrain dataset.  When the dataset delivers more or fewer
        channels (multispectral adaptation via ``_adapt_first_conv``), we
        tile the pretrained RGB mean/std to match — same pattern used by
        ``adapt_input_conv`` on the first conv weights — so the input conv
        and the normalize stay consistent.  Results on N != 3 channels
        should be marked as "adapted*" since both layers deviate from the
        canonical pretrain pipeline.

        Before applying the weights' Normalize we *also* convert the input
        to the scale the Normalize was calibrated for.  Without this, a
        reflectance-scaled dataset (e.g. so2sat in [0, 2.8]) hitting a
        weights' ``Normalize(mean=[0], std=[10000])`` becomes near-zero
        and the features collapse.
        """
        # Scale conversion: bring inputs into the scale the weights' Normalize
        # was calibrated for.  Required when a weights_normalize layer exists
        # (e.g. ResNet with Normalize(std=10000)) — without it a reflectance
        # dataset would produce near-zero outputs.  Also required for
        # model_native, which relies on this conversion explicitly.
        #
        # Skip when there is NO weights_normalize and strategy is not
        # model_native: the strategy (bandspec_zscore, identity, …) in
        # super().normalize_inputs already handles scaling correctly, and
        # applying unit conversion first would corrupt it (e.g. z-score uses
        # DN-scale mean/std — dividing raw DN by 10 000 before z-scoring
        # produces values ≈ 0 - 1000/500 ≈ -2, i.e. garbage).
        # The weights' own Normalize *is* the model-native pipeline, so it runs
        # only under model_native.  Applying it under every strategy made
        # dataset.normalization a no-op for each torchgeo model that ships a
        # transform, silently collapsing the normalization ablation.
        native = self.normalization is NormalizationStrategy.MODEL_NATIVE
        weights_norm = self._weights_normalize if native else None
        _need_unit_conv = self._weights_target_unit is not None and native
        if _need_unit_conv:
            images = convert_unit(images, self._dataset_input_unit, self._weights_target_unit)
        if weights_norm is not None:
            channel_counts: list[int] = []
            for m in weights_norm.modules():
                mean = getattr(m, "mean", None)
                if mean is None:
                    continue
                if isinstance(mean, torch.Tensor):
                    channel_counts.append(mean.shape[-1] if mean.ndim else mean.numel())
                else:
                    channel_counts.append(len(mean))
            in_chans = images.shape[1]
            if not channel_counts or all(count in (1, in_chans) for count in channel_counts):
                return weights_norm(images)
            # Channel count mismatch: build a tiled Normalize to match.
            tiled = self._tiled_normalize(in_chans)
            if tiled is not None:
                return tiled(images)
        return super().normalize_inputs(images)
