"""Module entry point: ``python -m torchgeo_bench`` runs the Hydra benchmark."""

from .main import main

if __name__ == "__main__":
    main()  # type: ignore[misc]  # pragma: no cover
