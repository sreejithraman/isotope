.PHONY: help install dev-setup lint format fix test typecheck clean

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  dev-setup   Install deps + pre-commit hooks (recommended)"
	@echo "  install     Install deps only"
	@echo ""
	@echo "Development:"
	@echo "  fix         Auto-fix lint/format issues"
	@echo "  lint        Check for lint/format issues"
	@echo "  format      Format code with ruff"
	@echo "  test        Run tests"
	@echo "  typecheck   Run mypy"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean       Remove build artifacts and caches"

install:
	pip install -e ".[dev,default]"

dev-setup: install
	pre-commit install
	@echo "Done! Pre-commit hooks will now run on every commit."

lint:
	ruff check src tests
	ruff format --check src tests

format:
	ruff format src tests

fix:
	ruff check --fix src tests
	ruff format src tests

test:
	pytest

typecheck:
	mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
