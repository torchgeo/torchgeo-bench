# Pass overrides as make arguments; SPHINX_BUILD and SPHINX_BUILD_OPTS also accept environment values.
SPHINX_BUILD      ?= sphinx-build
SPHINX_BUILD_OPTS ?= -W -j auto --keep-going
SOURCE_DIR        = .
BUILD_DIR         = _build

# Keep help as the default target.
help:
	@$(SPHINX_BUILD) -M help "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_BUILD_OPTS)

.PHONY: help Makefile

# Let Sphinx handle targets such as html and clean.
%: Makefile
	@$(SPHINX_BUILD) -M $@ "$(SOURCE_DIR)" "$(BUILD_DIR)" $(SPHINX_BUILD_OPTS)
