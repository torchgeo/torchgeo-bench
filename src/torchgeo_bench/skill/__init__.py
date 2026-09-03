"""Agent-facing usage instructions shipped with the package.

``SKILL.md`` is a self-contained brief that teaches a coding agent how to
drive ``torchgeo-bench``: which commands exist, how config overrides work,
where results land, and which repository conventions apply when extending
it.  ``torchgeo-bench --skill`` prints it so an agent can read (or save) it
without cloning the repository.
"""

from importlib.resources import files

SKILL_FILE = files(__name__) / "SKILL.md"


def read_skill() -> str:
    """Return the packaged ``SKILL.md`` instructions as text."""
    return SKILL_FILE.read_text(encoding="utf-8")
