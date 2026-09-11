"""Strict FP32 feature caches and decoder evaluation shared by the two studies."""

import hashlib
import json
import os
import platform
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise
from pathlib import Path
from typing import IO, Any

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.models.segmentation_heads import (
    ConvBlockHead,
    DPTHead,
    FPNHead,
    LinearHead,
    PatchLinearHead,
)
from torchgeo_bench.segmentation_probe import GPUTensorCache, SegmentationProbe

HEADS = ("linear", "conv_block", "fpn", "dpt", "patch_linear")
SPLITS = ("train", "val", "test")
IGNORE_INDEX = 255


class DivergedError(RuntimeError):
    """A nonfinite objective, gradient or buffer ended a trial."""


def identity(value: Any) -> str:
    """Hash JSON-compatible scientific configuration canonically."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    """Fingerprint names, shapes, dtypes and exact tensor bytes."""
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def software_versions() -> dict:
    """Identify installed dependencies and implementation, excluding session experiments."""
    import torchgeo_bench

    package = Path(torchgeo_bench.__file__).parent
    scripts = Path(__file__).parent
    paths = sorted(package.rglob("*.py")) + sorted(package.rglob("*.yaml"))
    paths += [
        scripts / name
        for name in (
            "_seg_sweep_common.py",
            "_segmentation_features.py",
            "_segmentation_convergence.py",
            "_segmentation_study.py",
            "run_segmentation_optimizer_study.py",
            "run_segmentation_layer_study.py",
        )
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(
            str(path.relative_to(package if path.is_relative_to(package) else scripts)).encode()
        )
        digest.update(path.read_bytes())
    packages = {
        name: version(name)
        for name in ("torch", "torchvision", "timm", "torchgeo", "geobenchv2", "numpy", "omegaconf")
    }
    try:
        packages["transformers"] = version("transformers")
    except PackageNotFoundError:  # allow-except: transformers is an optional dependency
        packages["transformers"] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "source_sha256": digest.hexdigest(),
    }


def hardware(device: torch.device) -> dict:
    """Record hardware separately from scientific identity, including scheduling index."""
    return {
        "device": str(device),
        "name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def seed_everything(seed: int) -> None:
    """Reset initialization and minibatch RNGs independently of LR and scheduling."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 -- Seed upstream global NumPy users as well.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_device(device: torch.device, *, strict: bool = True) -> None:
    """Require deterministic FP32 computation without AMP or TF32."""
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def synchronize(device: torch.device) -> None:
    """Wait for device work at timing boundaries."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def atomic_stream(path: Path, mode: str) -> Iterator[IO]:
    """Durably replace an artifact under the caller's exclusive output lock."""
    staging = path.with_name(path.name + ".writing")
    try:
        with staging.open(mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        staging.replace(path)
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if os.name != "nt" and directory_flag is not None:
            descriptor = os.open(path.parent, os.O_RDONLY | directory_flag)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        staging.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    """Write strict JSON without exposing partial state."""
    with atomic_stream(path, "w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_save(path: Path, value: Any) -> None:
    """Write a weights-only-loadable tensor artifact."""
    with atomic_stream(path, "wb") as stream:
        torch.save(value, stream)


@dataclass(frozen=True)
class ExtractionConfig:
    """Raw-input extraction controls; the layer union is supplied separately."""

    dataset: str = "burn_scars"
    model: str = "timm/vit/vit_small_patch16_224"
    bands: str = "rgb"
    image_size: int = 224
    normalization: str = "model_native"
    extract_batch_size: int = 16
    strict_determinism: bool = True

    def __post_init__(self) -> None:
        if self.bands not in ("rgb", "all"):
            raise ValueError("bands must be rgb or all.")
        if min(self.image_size, self.extract_batch_size) < 1:
            raise ValueError("Image and extraction batch sizes must be positive.")


def validate_layers(layers: list[str] | tuple[str, ...]) -> None:
    """Reject empty, malformed or repeated feature connections."""
    if not layers or any(not isinstance(name, str) or not name.strip() for name in layers):
        raise ValueError("Layers must be nonempty strings.")
    if len(set(layers)) != len(layers):
        raise ValueError("Duplicate layers are not allowed.")


def cache_spec(config: ExtractionConfig, layers: list[str]) -> dict:
    """Resolve extraction identity without loading data or pretrained weights."""
    validate_layers(layers)
    bench = get_bench_dataset_class(config.dataset)()
    if bench.task != "segmentation":
        raise ValueError("This study requires a segmentation dataset.")
    bands = bench.select_band_specs(bench.rgb_bands) if config.bands == "rgb" else bench.bands
    cfg = compose_config([f"model={config.model}"])
    return {
        "schema": 1,
        **asdict(config),
        "split_counts": dict(bench.split_sizes),
        "num_classes": bench.num_classes,
        "model_config": OmegaConf.to_container(cfg.model, resolve=True),
        "layers": layers,
        "band_specs": [asdict(band) for band in bands],
        "interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "partition": "default",
        "temporal_pool": "mean",
        "dtype": "float32",
        "extraction_seed": 0,
        "versions": software_versions(),
    }


def cache_tensors(splits: dict) -> dict[str, torch.Tensor]:
    """Flatten cache tensors for content fingerprinting."""
    return {
        f"{split}/{name}": tensor
        for split, data in splits.items()
        for name, tensor in [("masks", data["masks"]), *enumerate(data["features"])]
    }


def validate_cache(payload: dict, expected: dict) -> None:  # noqa: C901 -- Check all cache invariants.
    """Refuse stale configuration, truncated splits, invalid geometry or corrupt bytes."""
    metadata = payload["metadata"]
    validate_layers(expected["layers"])
    if metadata["spec"] != expected or metadata["key"] != identity(expected):
        raise ValueError("Cache configuration mismatch; use a new cache path.")
    if set(payload["splits"]) != set(SPLITS):
        raise ValueError("Cache must contain exactly train, val and test.")
    shapes = None
    for split in SPLITS:
        data = payload["splits"][split]
        masks, features = data["masks"], data["features"]
        n, size = expected["split_counts"][split], expected["image_size"]
        if masks.dtype != torch.int64 or tuple(masks.shape) != (n, size, size):
            raise ValueError(f"Invalid mask shape/dtype or split count in {split}.")
        allowed = ((masks >= 0) & (masks < expected["num_classes"])) | (masks == IGNORE_INDEX)
        if not allowed.all() or not (masks != IGNORE_INDEX).any():
            raise ValueError(f"Invalid or entirely ignored labels in {split}.")
        if len(features) != len(expected["layers"]):
            raise ValueError(f"Wrong feature layer count in {split}.")
        for tensor in features:
            if (
                tensor.dtype != torch.float32
                or tensor.ndim != 4
                or tensor.shape[0] != n
                or min(tensor.shape) < 1
                or tensor.requires_grad
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid FP32 feature tensor in {split}.")
        current = [list(t.shape[1:]) for t in features]
        if shapes is not None and shapes != current:
            raise ValueError("Feature geometry differs between splits.")
        shapes = current
    if metadata["feature_shapes"] != shapes:
        raise ValueError("Cache geometry metadata mismatch.")
    if tensor_digest(cache_tensors(payload["splits"])) != metadata["tensor_sha256"]:
        raise ValueError("Cache tensor checksum mismatch.")


def load_cache(path: Path, expected: dict) -> dict:
    """Load only tensors/primitive metadata and verify before use."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_cache(payload, expected)
    return payload


class FeatureExtractor(SegmentationProbe):
    """Use probe hook/token handling without its fixed-224 dummy forward or decoder."""

    def __init__(self, backbone: BenchModel, layers: list[str]) -> None:
        nn.Module.__init__(self)
        self.backbone, self.layer_names = backbone, layers
        self.temporal_pool = "mean"
        self._features = {}
        self.hooks = []
        self.register_hooks()

    @torch.no_grad()
    def extract(self, loader: DataLoader) -> dict:
        """Encode each actual input once, optionally averaging temporal features."""
        features: list[list[torch.Tensor]] = [[] for _ in self.layer_names]
        masks = []
        for batch in loader:
            if isinstance(batch, dict):
                image, mask = batch["image"], batch["mask"]
            else:
                image, mask = batch[:2]
            image = image.to(device=self._backbone_device(), dtype=torch.float32)
            steps = image.shape[1] if image.ndim == 5 else 0
            if steps:
                image = image.flatten(0, 1)
            self._features.clear()
            self.backbone(image)
            for name, batches in zip(self.layer_names, features, strict=True):
                value = self._process_feature(self._features[name])
                if steps:
                    value = self._pool_time(value, steps)
                batches.append(value.detach().to(device="cpu", dtype=torch.float32))
            if mask.ndim == 4 and mask.shape[1] == 1:
                mask = mask.squeeze(1)
            masks.append(mask.long().cpu())
        return {"features": [torch.cat(batches) for batches in features], "masks": torch.cat(masks)}


def prepare_cache(path: Path, spec: dict, device: torch.device, num_workers: int) -> dict:
    """Extract complete official splits once through a frozen BenchModel wrapper."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock", timeout=0):
        if path.exists():
            return load_cache(path, spec)
        configure_device(device, strict=spec["strict_determinism"])
        seed_everything(spec["extraction_seed"])
        start = time.perf_counter()
        bench = get_bench_dataset_class(spec["dataset"])()
        bands = bench.select_band_specs(bench.rgb_bands) if spec["bands"] == "rgb" else bench.bands
        _, train, val, test = get_datasets(
            dataset_name=spec["dataset"],
            return_val=True,
            batch_size=spec["extract_batch_size"],
            num_workers=num_workers,
            image_size=spec["image_size"],
            interpolation=spec["interpolation"],
            bands=spec["bands"],
        )
        model = instantiate(spec["model_config"], bands=bands, normalization=spec["normalization"])
        if not isinstance(model, BenchModel):
            raise TypeError("The model configuration must instantiate a BenchModel.")
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        probe = FeatureExtractor(model, spec["layers"])
        metadata = {
            "spec": spec,
            "key": identity(spec),
            "backbone_state_sha256": tensor_digest(model.state_dict()),
            "hardware": hardware(device),
            "extraction_seconds": {},
        }
        synchronize(device)
        metadata["setup_seconds"] = time.perf_counter() - start
        splits = {}
        try:
            for split, loader in zip(SPLITS, (train, val, test), strict=True):
                if len(loader.dataset) != spec["split_counts"][split]:
                    raise ValueError(f"{split} must use the entire official split.")
                ordered = DataLoader(
                    loader.dataset,
                    batch_size=spec["extract_batch_size"],
                    shuffle=False,
                    num_workers=num_workers,
                    collate_fn=loader.collate_fn,
                )
                synchronize(device)
                start = time.perf_counter()
                splits[split] = probe.extract(ordered)
                synchronize(device)
                metadata["extraction_seconds"][split] = time.perf_counter() - start
        finally:
            for hook in probe.hooks:
                hook.remove()
        metadata["feature_shapes"] = [list(t.shape[1:]) for t in splits["train"]["features"]]
        metadata["tensor_sha256"] = tensor_digest(cache_tensors(splits))
        payload = {"metadata": metadata, "splits": splits}
        validate_cache(payload, spec)
        atomic_save(path, payload)
        return payload


def select_layers(payload: dict, layers: list[str]) -> tuple[dict, dict]:
    """Select a named ordered subset without copying tensors or losing union provenance."""
    validate_layers(layers)
    metadata = payload["metadata"]
    available = metadata["spec"]["layers"]
    if not set(layers) <= set(available):
        raise ValueError("Requested layer is absent from the union cache.")
    indices = [available.index(name) for name in layers]
    selected = {
        split: {"masks": data["masks"], "features": [data["features"][i] for i in indices]}
        for split, data in payload["splits"].items()
    }
    return selected, {
        **metadata,
        "selected_layers": layers,
        "selected_feature_shapes": [metadata["feature_shapes"][i] for i in indices],
    }


def device_cache(data: dict, device: torch.device) -> GPUTensorCache:
    """Keep resident features in FP32, including on CUDA."""
    return GPUTensorCache(
        [t.to(device=device, dtype=torch.float32).contiguous() for t in data["features"]],
        data["masks"].to(device=device, dtype=torch.long).contiguous(),
        device,
    )


def check_head_layers(head: str, layers: list[str] | tuple[str, ...]) -> None:
    """Refuse incompatible connections instead of ignoring layers or changing heads."""
    validate_layers(layers)
    if head not in HEADS:
        raise ValueError(f"Unknown head {head}.")
    if head == "dpt" and len(layers) != 4:
        raise ValueError("DPT requires exactly four layers.")
    if head == "patch_linear" and len(layers) != 1:
        raise ValueError("Patch-linear requires exactly one layer.")


def preflight_heads(heads: list[str]) -> None:
    """Fail before extraction if a requested decoder's optional dependency is unavailable."""
    if "dpt" not in heads:
        return
    try:
        module = import_module("transformers.models.dpt.modeling_dpt")
    except ModuleNotFoundError as error:  # allow-except: explain a missing optional dependency
        if error.name == "transformers":
            raise ModuleNotFoundError(
                "The dpt head requires the optional transformers dependency; "
                "install it or choose --heads without dpt."
            ) from error
        raise
    if not hasattr(module, "DPTFeatureFusionLayer"):
        raise ImportError("Installed transformers lacks the DPTFeatureFusionLayer decoder API.")


def make_head(name: str, cache: GPUTensorCache, num_classes: int, hidden_dim: int) -> nn.Module:
    """Materialize lazy parameters using the actual selected feature/mask geometry."""
    check_head_layers(name, [str(i) for i in range(len(cache.layer_tensors))])
    shapes = [tuple(t.shape[-2:]) for t in cache.layer_tensors]
    if name in ("fpn", "dpt") and any(h > nh or w > nw for (h, w), (nh, nw) in pairwise(shapes)):
        raise ValueError("FPN/DPT features must be ordered coarse-to-fine.")
    constructors = {
        "linear": LinearHead,
        "conv_block": ConvBlockHead,
        "fpn": FPNHead,
        "dpt": DPTHead,
        "patch_linear": PatchLinearHead,
    }
    kwargs = {} if name in ("linear", "patch_linear") else {"hidden_dim": hidden_dim}
    head = constructors[name]([t.shape[1] for t in cache.layer_tensors], num_classes, **kwargs)
    head.to(device=cache.device, dtype=torch.float32).eval()
    with torch.no_grad():
        head([t[:1].clone() for t in cache.layer_tensors], *cache.masks.shape[-2:])
    return head.train()


def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy parameters and persistent normalization buffers to CPU."""
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def require_finite(tensors: Iterator[torch.Tensor] | list[torch.Tensor], stage: str) -> None:
    """Raise a specific divergence error rather than publish NaNs."""
    if any(not torch.isfinite(tensor).all() for tensor in tensors):
        raise DivergedError(f"Nonfinite {stage}.")


def summed_cross_entropy(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Use deterministic 2-D CE, avoiding CUDA's atomic NCHW reduction."""
    pixels = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])
    return F.cross_entropy(pixels, masks.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum")


def reject_stochastic_modules(head: nn.Module) -> None:
    """Reject stochastic activations incompatible with deterministic closures."""
    for module in head.modules():
        if isinstance(
            module,
            (nn.modules.dropout._DropoutNd, nn.AlphaDropout, nn.FeatureAlphaDropout, nn.RReLU),
        ) or type(module).__name__ in ("DropPath", "StochasticDepth"):
            raise ValueError(f"Closures cannot use stochastic module {type(module).__name__}.")


@torch.no_grad()
def recalibrate_bn(head: nn.Module, cache: GPUTensorCache, batch_size: int) -> bool:
    """Reset and cumulatively recalibrate BN on ordered training images only."""
    modules = [m for m in head.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not modules:
        return False
    momenta = [m.momentum for m in modules]
    head.train()
    try:
        for module in modules:
            module.reset_running_stats()
            module.momentum = None
        for features, masks in cache.ordered_batches(batch_size):
            require_finite([head(features, *masks.shape[-2:])], "BN calibration")
        require_finite(iter(head.buffers()), "BN buffers")
    finally:
        for module, momentum in zip(modules, momenta, strict=True):
            module.momentum = momentum
    return True


def per_image_confusions(
    predictions: torch.Tensor, masks: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Return (N,C,C) true-row/predicted-column counts excluding ignored labels."""
    valid = (masks >= 0) & (masks < num_classes) & (masks != IGNORE_INDEX)
    offsets = torch.arange(len(masks), device=masks.device)[:, None, None] * num_classes**2
    indices = (offsets + masks * num_classes + predictions)[valid]
    return torch.bincount(indices, minlength=len(masks) * num_classes**2).reshape(
        len(masks), num_classes, num_classes
    )


def macro_miou(confusion: torch.Tensor) -> float:
    """Macro-average IoU, assigning zero to zero-denominator classes."""
    matrix = confusion.double()
    intersection = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - intersection
    return float((intersection / union.clamp_min(1)).mean())


@torch.no_grad()
def evaluate(head: nn.Module, cache: GPUTensorCache, batch_size: int) -> dict:
    """Evaluate a frozen head without updating BN."""
    head.eval()
    confusions = []
    loss_sum = torch.zeros((), dtype=torch.float64, device=cache.device)
    for features, masks in cache.ordered_batches(batch_size):
        logits = head(features, *masks.shape[-2:])
        require_finite([logits], "evaluation logits")
        loss_sum += summed_cross_entropy(logits, masks)
        confusions.append(per_image_confusions(logits.argmax(1), masks, logits.shape[1]).cpu())
    require_finite([loss_sum], "evaluation loss")
    per_image = torch.cat(confusions)
    return {
        "miou": macro_miou(per_image.sum(0)),
        "ce": float(loss_sum / cache.masks.ne(IGNORE_INDEX).sum()),
        "per_image_confusions": per_image,
    }


def bootstrap_interval(confusions: torch.Tensor, samples: int, seed: int) -> list[float]:
    """Return deterministic image-bootstrap 95% CIs, not seed-variance intervals."""
    generator = torch.Generator().manual_seed(seed)
    scores = [
        macro_miou(
            confusions[torch.randint(len(confusions), (len(confusions),), generator=generator)].sum(
                0
            )
        )
        for _ in range(samples)
    ]
    return torch.quantile(
        torch.tensor(scores, dtype=torch.float64),
        torch.tensor([0.025, 0.975], dtype=torch.float64),
    ).tolist()
