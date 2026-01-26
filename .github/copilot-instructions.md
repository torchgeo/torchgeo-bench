---
applyTo: "**/*.py, **/*.ipynb"
---
# Project general coding standards for Python

- Always `conda activate torchgeo-bench` before running commands
- Assume Python 3.12+ and Pydantic v2.0
- Prefer modern type hints (e.g., `list[str]` instead of `List[str]`)
- Use `logging` for logging, not `print()`
- Don't create documentation for refactoring
 