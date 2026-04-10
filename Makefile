# Makefile for torchgeo-bench

UV := uv run

.PHONY: install sync tests lint format clean help

install:
	uv sync --all-extras

sync:
	$(MAKE) install

tests:
	$(UV) pytest

lint:
	$(UV) pre-commit run --all-files

format:
	$(UV) ruff format src/ tests/
	$(UV) ruff check --fix --select I src/ tests/

clean:
	rm -rf htmlcov .pytest_cache .coverage

help:
	@echo "Available targets:"
	@echo "  install - Install dependencies and dev tools with uv"
	@echo "  sync    - Install dependencies and dev tools with uv"
	@echo "  tests   - Run test suite with coverage"
	@echo "  lint    - Run pre-commit checks on all files"
	@echo "  format  - Format code and auto-fix imports with ruff"
	@echo "  clean   - Remove generated files (htmlcov, .coverage, .pytest_cache)"
	@echo "  help    - Show this help message"
