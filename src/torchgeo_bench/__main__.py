"""Module entry point for ``python -m torchgeo_bench``.

Sets up Hydra config search paths so that:
  1. A ``conf/`` directory in the CWD (if present) is searched **first**.
  2. The configs bundled inside the installed package serve as a fallback.

This means users can override any built-in YAML (including ``config.yaml``
and individual model configs) simply by placing files in a local ``conf/``
directory, while all unoverridden configs are still found in the package.
"""

import os
import sys


def _setup_config_paths() -> None:
    """Configure Hydra search paths: CWD ``conf/`` first, then package.

    With ``config_path=None`` in the decorator, the ``main`` provider is
    empty.  We populate the search path entirely via ``--config-dir`` (for
    the highest-priority source) and ``hydra.searchpath`` (for fallbacks).

    Only injects flags when the user has **not** already passed
    ``--config-dir`` or ``--config-path``.
    """
    user_overrode = any(
        a.startswith(("--config-dir", "--config-path")) for a in sys.argv
    )
    if user_overrode:
        return

    cwd_conf = os.path.join(os.getcwd(), "conf")
    if os.path.isdir(cwd_conf):
        # Replace the decorator's config_path with CWD conf (highest priority),
        # and add packaged configs as a fallback search path.
        sys.argv.insert(1, f"--config-path={cwd_conf}")
        sys.argv.append("hydra.searchpath=[pkg://torchgeo_bench.conf]")


if __name__ == "__main__":
    _setup_config_paths()
    from .main import main

    main()  # type: ignore[misc]
