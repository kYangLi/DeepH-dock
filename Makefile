.PHONY: clean install test build help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip3

.DEFAULT_GOAL := help

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install    Create .venv with uv and install package in editable mode"
	@echo "  test       Run all tests"
	@echo "  build      Build distribution wheel"
	@echo "  clean      Remove build artifacts and cache files"
	@echo "  help       Show this help message"

install:
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment with uv..."; \
		uv venv; \
	fi
	@echo "Installing package in editable mode with dev dependencies..."
	uv pip install -e .[dev]

test:
	bash tests/run_test_all.sh

build:
	uv build --wheel -o dist ./

clean:
	@echo "Cleaning build artifacts and cache files..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage dist build 2>/dev/null || true
	@echo "Clean complete!"
