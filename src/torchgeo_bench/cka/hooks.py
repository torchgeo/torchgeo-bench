"""Hook-based activation collection for CKA analysis."""

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path, supporting integer indexing.

    Args:
        model: Root module.
        path: Dotted submodule path (for example ``"backbone.blocks.11"``).

    Returns:
        The resolved submodule.

    Raises:
        AttributeError: If an attribute path component is missing.
        IndexError: If an integer index is out of range.
        TypeError: If indexing is attempted on a non-indexable module.
    """
    module: nn.Module | nn.ModuleList | nn.Sequential = model
    for part in path.split("."):
        if part.isdigit():
            idx = int(part)
            if isinstance(module, (nn.ModuleList, nn.Sequential, list, tuple)):
                module = module[idx]  # type: ignore[index]
            else:
                raise TypeError(f"Cannot index into module {type(module).__name__} with component {part!r}.")
        else:
            if not hasattr(module, part):
                raise AttributeError(f"Module path {path!r} is invalid at component {part!r}.")
            module = getattr(module, part)
    if not isinstance(module, nn.Module):
        raise TypeError(f"Resolved object at path {path!r} is not an nn.Module: {type(module)}")
    return module


class HookCollector:
    """Collect pooled activations from named forward-hook locations."""

    def __init__(self, model: nn.Module, hook_paths: list[str]) -> None:
        if not hook_paths:
            raise ValueError("hook_paths must be non-empty.")
        self.model = model
        self.hook_paths = list(hook_paths)
        self._buffers: dict[str, list[np.ndarray]] = defaultdict(list)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

        for path in self.hook_paths:
            module = _resolve_module(model, path)
            handle = module.register_forward_hook(self._make_hook(path))
            self._handles.append(handle)

    def _make_hook(self, path: str):
        def _hook(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Hook path {path!r} produced non-tensor output: {type(output)}")
            pooled = self._pool_output(output)
            self._buffers[path].append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))

        return _hook

    @staticmethod
    def _pool_output(output: torch.Tensor) -> torch.Tensor:
        if output.ndim == 3:
            return output.mean(dim=1)
        if output.ndim == 4:
            return output.mean(dim=(-2, -1))
        if output.ndim == 2:
            return output
        raise ValueError(f"Unsupported hook output shape: {tuple(output.shape)}")

    def collect(self) -> dict[str, np.ndarray]:
        """Return pooled activations collected since the last call.

        Returns:
            Mapping of hook path to array ``(N, D)``. Paths that have not seen
            activations return an empty ``(0, 0)`` array.
        """
        out: dict[str, np.ndarray] = {}
        for path in self.hook_paths:
            chunks = self._buffers[path]
            if not chunks:
                out[path] = np.empty((0, 0), dtype=np.float32)
            else:
                out[path] = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
            self._buffers[path] = []
        return out

    def remove(self) -> None:
        """Deregister all hook handles."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "HookCollector":
        return self

    def __exit__(self, *_: object) -> None:
        self.remove()
