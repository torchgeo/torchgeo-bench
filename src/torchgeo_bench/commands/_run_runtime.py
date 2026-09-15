"""Execute validated image settings without adapting the runtime configuration."""

from ..config.run import RunConfig
from ..main import main


def run(config: RunConfig) -> None:
    """Pass typed settings to the image runner, which validates the execution device."""
    main(config, strict=True)
