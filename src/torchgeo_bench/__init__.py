"""Public package exports for torchgeo-bench.

Heavy submodules (torch, torchgeo, sklearn) load lazily so importing the
package — and therefore CLI startup — stays fast.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torchgeo-bench")
except PackageNotFoundError:  # allow-except: source-only imports before package installation
    __version__ = "0.5.0"

__author__ = "torchgeo-bench contributors"

__all__: list[str] = [
    "bootstrap_map",
    "evaluate_knn",
    "evaluate_logistic",
]


def __getattr__(name: str) -> object:
    if name in __all__:
        from torchgeo_bench import main

        return getattr(main, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
