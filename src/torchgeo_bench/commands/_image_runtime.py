"""Execute validated image settings without adapting the runtime configuration."""

from ..config_schema import RunConfig
from ..main import main, resolve_image_device


def run(config: RunConfig) -> None:
    """Resolve the execution device and pass typed settings to the image runner."""
    device = resolve_image_device(config.runtime.device)
    runtime = config.runtime.model_copy(update={"device": str(device)})
    main(config.model_copy(update={"runtime": runtime}), strict=True)
