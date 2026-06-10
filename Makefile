# Makefile for torchgeo-bench

CONDA_RUN := conda run --no-capture-output -n torchgeo-bench

.PHONY: install sync tests lint format clean docs docs-clean help

install:
	conda env update -n torchgeo-bench -f environment.yml
	$(CONDA_RUN) pip install -e ".[dev]"

sync:
	$(MAKE) install

tests:
	$(CONDA_RUN) pytest

lint:
	$(CONDA_RUN) pre-commit run --all-files

format:
	$(CONDA_RUN) ruff format src/ tests/
	$(CONDA_RUN) ruff check --fix --select I src/ tests/

docs:
	sphinx-build -b html docs/ docs/_build/html

docs-clean:
	rm -rf docs/_build

clean:
	rm -rf htmlcov .pytest_cache .coverage

help:
	@echo "Available targets:"
	@echo "  install - Install dependencies into torchgeo-bench conda env"
	@echo "  sync    - Alias for install"
	@echo "  tests   - Run test suite with coverage"
	@echo "  lint    - Run pre-commit checks on all files"
	@echo "  format  - Format code and auto-fix imports with ruff"
	@echo "  docs       - Build HTML documentation into docs/_build/html"
	@echo "  docs-clean - Remove the docs build directory"
	@echo "  clean      - Remove generated files (htmlcov, .coverage, .pytest_cache)"
	@echo "  help       - Show this help message"
