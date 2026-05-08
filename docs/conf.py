# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""Sphinx configuration for the torchgeo-bench documentation site."""

# -- Path setup --------------------------------------------------------------

import inspect
import os
import sys

# Make the src/ layout importable so autodoc can resolve the package without
# requiring it to be installed (the GitHub Pages workflow installs the package
# via 'uv sync --extra docs', but a bare 'make html' from a fresh checkout
# should still work).
sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

import torchgeo_bench

# -- Project information -----------------------------------------------------

project = "torchgeo-bench"
copyright = "torchgeo-bench Contributors"
author = torchgeo_bench.__author__
version = ".".join(torchgeo_bench.__version__.split(".")[:2])
release = torchgeo_bench.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

# Files / directories the builder should ignore.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "plans"]

# Sphinx 5.3+ required to allow section titles inside autodoc class docstrings
# https://github.com/sphinx-doc/sphinx/pull/10887
needs_sphinx = "5.3"

# Surface every cross-reference miss as a warning so the RTD build (with
# ``fail_on_warning: true``) will fail loudly on broken docs.  Known
# unfixable misses go in ``nitpick_ignore`` below.
nitpicky = True
nitpick_ignore = [
    # Private internal types referenced from public docstrings / autoclass
    # ``:show-inheritance:`` (intentionally not exported).
    ("py:class", "torchgeo_bench.datasets.geobench_v1._V1Dataset"),
    ("py:class", "torchgeo_bench.datasets.geobench_v2._V2Dataset"),
    ("py:class", "torchgeo_bench.models.torchgeo_models._TorchGeoBackboneBench"),
    ("py:class", "torchgeo_bench.segmentation_probe.CachedFeaturesDataset"),
    ("py:class", "torchgeo_bench.segmentation_probe.GPUTensorCache"),
    ("py:class", "CachedFeaturesDataset"),
    ("py:class", "GPUTensorCache"),
    # Source-docstring references to module-level constants / private helpers
    ("py:data", "_REGISTRY"),
    ("py:data", "OLMOEARTH_S2_BANDS"),
    ("py:data", "DEFAULT_V2_DATASETS"),
    ("py:class", "Normalizer"),
    ("py:class", "RCF"),
    ("py:class", "EvaluationResult"),
    ("py:class", "BandSpec"),
    ("py:class", "GeoBenchv2"),
    ("py:class", "Single-label"),
    ("py:class", "auto_resize /"),
    ("py:class", "FeatureFusionBlock"),
    ("py:attr", "BenchModel.bands"),
    ("py:attr", "num_channels"),
    ("py:attr", "base.BenchDataset.supports_partitions"),
    ("py:meth", "_forward_patch_features"),
    ("py:meth", "SegmentationProbe.extract_segmentation_features"),
    ("py:mod", "torchgeo_bench.main"),
    # Third-party types we don't control intersphinx mappings for
    ("py:class", "faiss.swigfaiss_avx2.IndexFlatL2"),
    ("py:class", "geobench.task.Task"),
    ("py:class", "geobench_v2.GeoBenchDataModule"),
    ("py:class", "geobenchv2.task.Task"),
    ("py:class", "h5py._hl.dataset.Dataset"),
    ("py:class", "hydra.core.config_store.ConfigStore"),
    ("py:class", "omegaconf.dictconfig.DictConfig"),
    ("py:class", "omegaconf.listconfig.ListConfig"),
    ("py:class", "timm.models.resnet.ResNet"),
    ("py:class", "timm.models.vision_transformer.VisionTransformer"),
    ("py:class", "torchgeo.models.api.WeightsEnum"),
    ("py:class", "transformers.modeling_utils.PreTrainedModel"),
    # Generic / forward references we resolve at runtime
    ("py:class", "Self"),
]

# Modules whose entries are looked up via intersphinx — when intersphinx is
# unreachable (e.g. an offline build), don't fail the build over them.  When
# the network is available (CI, GitHub Pages) the inventory fetches succeed
# and these references resolve correctly.
nitpick_ignore_regex = [
    (r"py:.*", r"^(numpy|torch|sklearn|pandas|matplotlib|PIL|pillow|torchgeo|torchvision)(\..*)?$"),
    (r"py:.*", r"^(pathlib|abc|collections|typing|argparse)(\..*)?$"),
]


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "collapse_navigation": False,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "navbar_align": "left",
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/torchgeo/torchgeo-bench",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "torchgeo",
    "github_repo": "torchgeo-bench",
    "github_version": "main",
    "doc_path": "docs",
}

html_static_path = ["_static"]
html_css_files = ["custom.css"]


# -- Extension configuration -------------------------------------------------

# sphinx.ext.autodoc
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "torchgeo": ("https://torchgeo.readthedocs.io/en/stable/", None),
    "torchvision": ("https://docs.pytorch.org/vision/stable/", None),
}

# myst-parser
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "smartquotes",
]
myst_heading_anchors = 3
suppress_warnings = ["myst.header", "myst.xref_missing", "misc.highlighting_failure"]

# sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


# sphinx.ext.linkcode
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Resolve a GitHub URL for the given Python object."""
    if domain != "py":
        return None

    modname = info.get("module", "")
    fullname = info.get("fullname", "")
    if not modname:
        return None

    try:
        mod = sys.modules.get(modname)
        if mod is None:
            __import__(modname)
            mod = sys.modules[modname]

        obj = mod
        for part in fullname.split("."):
            obj = getattr(obj, part)

        obj = inspect.unwrap(obj)
        sourcefile = inspect.getsourcefile(obj)
        if sourcefile is None:
            return None
        source, lineno = inspect.getsource(obj), inspect.getsourcelines(obj)[1]
    except Exception:
        return None

    sourcefile = os.path.relpath(sourcefile, start=os.path.join(os.path.dirname(__file__), ".."))

    lineend = lineno + source.count("\n") - 1
    return f"https://github.com/torchgeo/torchgeo-bench/blob/main/{sourcefile}#L{lineno}-L{lineend}"
