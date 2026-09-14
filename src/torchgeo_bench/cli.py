"""Canonical flags-and-YAML command line interface."""


def main(argv: list[str] | None = None) -> None:
    """Dispatch the same commands as the installed console entry point."""
    from .image_cli import main as dispatch

    dispatch(argv)


if __name__ == "__main__":
    main()
