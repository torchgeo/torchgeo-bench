"""Regression tests for the argparse/Python 3.14 compatibility shim."""

import argparse

from torchgeo_bench._argparse_compat import patch_argparse_help


class _LazyHelp:
    """Mimics Hydra's lazily rendered, non-string ``help`` object."""

    def __repr__(self) -> str:
        return "Install or Uninstall shell completion"


def test_patch_allows_non_string_help() -> None:
    # On Python 3.14 add_argument eagerly validates help and raises
    # "badly formed help string" for a non-string help object; the shim coerces
    # it to str so the parser builds.  On older Pythons add_argument never
    # validated help, so this passes trivially.
    patch_argparse_help()

    parser = argparse.ArgumentParser()
    parser.add_argument("--shell-completion", action="store_true", help=_LazyHelp())

    assert parser.parse_args([]).shell_completion is False


def test_patch_is_idempotent() -> None:
    patch_argparse_help()
    patched = argparse.HelpFormatter._expand_help
    patch_argparse_help()
    assert argparse.HelpFormatter._expand_help is patched


def test_patch_preserves_string_help() -> None:
    patch_argparse_help()
    parser = argparse.ArgumentParser()
    parser.add_argument("--foo", help="plain string help")
    assert "plain string help" in parser.format_help()


def test_hydra_args_parser_builds() -> None:
    # The real-world entry point: Hydra registers --shell-completion with a
    # non-string help object, which is what breaks `torchgeo-bench run` on 3.14.
    patch_argparse_help()
    from hydra._internal.utils import get_args_parser

    parser = get_args_parser()
    namespace = parser.parse_args(["model=rcf", "dataset.names=[m-eurosat]"])
    assert namespace.overrides == ["model=rcf", "dataset.names=[m-eurosat]"]
