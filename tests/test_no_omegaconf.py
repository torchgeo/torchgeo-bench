"""Keep application code and maintained callers independent of OmegaConf."""

import ast
from pathlib import Path

import pytest


def omega_import_lines(source: str) -> list[int]:
    """Find direct and literal dynamic imports without importing their dependencies."""
    tree = ast.parse(source)
    lines = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            dynamic = (
                isinstance(function, ast.Name) and function.id in {"__import__", "import_module"}
            ) or (isinstance(function, ast.Attribute) and function.attr == "import_module")
            argument = node.args[0]
            if dynamic and isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                modules = [argument.value]
        if any(module.split(".")[0] == "omegaconf" for module in modules):
            lines.append(node.lineno)
    return sorted(lines)


@pytest.mark.parametrize(
    "source",
    [
        "import omegaconf",
        "import omegaconf as config",
        "import os, omegaconf.dictconfig",
        "from omegaconf import OmegaConf",
        "from omegaconf.dictconfig import DictConfig as Config",
        "def load():\n    from omegaconf import OmegaConf",
        "__import__('omegaconf')",
        "importlib.import_module('omegaconf.dictconfig')",
        "import_module('omegaconf')",
    ],
)
def test_import_guard_detects_application_dependencies(source: str) -> None:
    assert omega_import_lines(source)


@pytest.mark.parametrize(
    "source",
    [
        "assert 'omegaconf' not in sys.modules",
        "stored_hash = {'_target_': 'old.Model'}",
        "from third_party import Model",
        "import torchgeo",
        "from .omegaconf import local_symbol",
        "message = 'from omegaconf import OmegaConf'",
    ],
)
def test_import_guard_allows_assertions_fixtures_and_third_party_dependencies(source: str) -> None:
    assert not omega_import_lines(source)


@pytest.mark.parametrize(
    "directory", ["src", "tests", "examples", "experiments", "scripts", "projects"]
)
def test_no_application_owned_omegaconf_imports(directory: str) -> None:
    root = Path(__file__).parents[1]
    paths = sorted(
        path
        for path in (root / directory).rglob("*")
        if path.suffix in {".py", ".pyi"}
        and not any(part in {".venv", "__pycache__", "site-packages"} for part in path.parts)
    )
    assert paths, f"No maintained Python sources found under {directory}"
    violations = [
        f"{path.relative_to(root)}:{line}"
        for path in paths
        for line in omega_import_lines(path.read_text(encoding="utf-8"))
    ]
    assert not violations, "Application-owned OmegaConf imports:\n" + "\n".join(violations)
