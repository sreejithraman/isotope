.PHONY: help install dev-setup lint format fix test typecheck ci clean build release cli example ingest query

ORANGE := \033[38;5;208m
BOLD := \033[1m
DIM := \033[2m
RESET := \033[0m

help:
	@printf "\n"
	@printf "$(ORANGE)⛬ isotope$(RESET)\n"
	@printf "\n"
	@printf "$(DIM)╭─$(RESET)$(ORANGE) Setup $(RESET)$(DIM)────────────────────────────────────╮$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)dev-setup$(RESET)    Install + pre-commit hooks   $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)install$(RESET)      Install deps only            $(DIM)│$(RESET)\n"
	@printf "$(DIM)╰────────────────────────────────────────────╯$(RESET)\n"
	@printf "\n"
	@printf "$(DIM)╭─$(RESET)$(ORANGE) Development $(RESET)$(DIM)──────────────────────────────╮$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)fix$(RESET)          Auto-fix lint/format issues  $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)lint$(RESET)         Check for issues             $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)format$(RESET)       Format code with ruff        $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)test$(RESET)         Run tests                    $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)typecheck$(RESET)    Run mypy                     $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)ci$(RESET)           Run all checks (like CI)     $(DIM)│$(RESET)\n"
	@printf "$(DIM)╰────────────────────────────────────────────╯$(RESET)\n"
	@printf "\n"
	@printf "$(DIM)╭─$(RESET)$(ORANGE) Running $(RESET)$(DIM)────────────────────────────────────────────────╮$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)example$(RESET)      Install deps for examples (run this first!)      $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)cli$(RESET)          Run CLI: make cli ARGS=\"config\"                 $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)ingest$(RESET)       Ingest docs: make ingest [PATH_ARG=./path]      $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)query$(RESET)        Query: make query ARGS=\"'your question'\"        $(DIM)│$(RESET)\n"
	@printf "$(DIM)╰────────────────────────────────────────────────────────────╯$(RESET)\n"
	@printf "\n"
	@printf "$(DIM)╭─$(RESET)$(ORANGE) Release $(RESET)$(DIM)──────────────────────────────────╮$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)build$(RESET)        Build distribution packages  $(DIM)│$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)release$(RESET)      Tag and push a release       $(DIM)│$(RESET)\n"
	@printf "$(DIM)╰────────────────────────────────────────────╯$(RESET)\n"
	@printf "\n"
	@printf "$(DIM)╭─$(RESET)$(ORANGE) Cleanup $(RESET)$(DIM)──────────────────────────────────╮$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)clean$(RESET)        Remove build artifacts       $(DIM)│$(RESET)\n"
	@printf "$(DIM)╰────────────────────────────────────────────╯$(RESET)\n"

install:
	uv sync --extra dev --extra all

dev-setup: install
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push
	@echo "Done! Pre-commit hooks installed (lint on commit, mypy+pytest on push)."

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests

fix:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest

typecheck:
	uv run mypy src

ci: lint typecheck test

build:
	uv build

release: lint test build
ifndef VERSION
	$(error Usage: make release VERSION=0.1.0)
endif
	@# Check working tree is clean
	@git diff --quiet || (echo "Error: Uncommitted changes in working tree" && exit 1)
	@git diff --cached --quiet || (echo "Error: Staged changes not committed" && exit 1)
	@# Check tag doesn't already exist
	@if git show-ref --verify --quiet "refs/tags/v$(VERSION)"; then echo "Error: Tag v$(VERSION) already exists"; exit 1; fi
	@# Verify VERSION matches pyproject.toml
	@uv run python -c "import tomllib; v=tomllib.load(open('pyproject.toml','rb'))['project']['version']; exit(0 if v=='$(VERSION)' else print(f'Error: VERSION=$(VERSION) but pyproject.toml has {v}') or 1)"
	@echo "Creating release v$(VERSION)..."
	git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	git push origin "v$(VERSION)"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info .venv
	rm -f .install-cli .install-example
	find . -type d -name __pycache__ -exec rm -rf {} +

# Smart install markers - reinstall when pyproject.toml changes
.install-cli: pyproject.toml
	uv sync --extra cli
	@touch .install-cli

cli: .install-cli
	uv run isotope $(ARGS)

.install-example: pyproject.toml
	uv sync --extra cli --extra chroma --extra litellm --extra loaders
	@touch .install-example

example: .install-example
	@printf "$(ORANGE)✓ Example environment ready!$(RESET)\n"
	@printf "\nNext steps:\n"
	@printf "  1. uv run isotope init                              # Set up provider\n"
	@printf "  2. uv run isotope ingest examples/data/hacker-laws.pdf\n"
	@printf "  3. uv run isotope query 'What is Brooks Law?'\n"

# Load .env if present (created by `isotope init`)
-include .env
export ISOTOPE_LLM_API_KEY
export ISOTOPE_EMBEDDING_API_KEY

# Dev testing - uses Gemini Flash (free tier) by default
ingest: .install-example
ifndef ISOTOPE_LLM_API_KEY
	$(error Set ISOTOPE_LLM_API_KEY or run 'isotope init' first)
endif
	ISOTOPE_LITELLM_LLM_MODEL=gemini/gemini-3-flash-preview \
	ISOTOPE_LITELLM_EMBEDDING_MODEL=gemini/gemini-embedding-001 \
	isotope ingest --force $(or $(PATH_ARG),./docs)

	query: .install-example
	ifndef ISOTOPE_LLM_API_KEY
		$(error Set ISOTOPE_LLM_API_KEY or run 'isotope init' first)
	endif
		ISOTOPE_LITELLM_LLM_MODEL=gemini/gemini-3-flash-preview \
		ISOTOPE_LITELLM_EMBEDDING_MODEL=gemini/gemini-embedding-001 \
		isotope query $(ARGS)
