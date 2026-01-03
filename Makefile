.PHONY: help install dev-setup lint format fix test typecheck clean

ORANGE := \033[38;5;208m
BOLD := \033[1m
DIM := \033[2m
RESET := \033[0m

help:
	@printf "$(ORANGE)◆ isotope$(RESET)\n"
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
	@printf "$(DIM)╰────────────────────────────────────────────╯$(RESET)\n"
	@printf "\n"
	@printf "$(DIM)╭─$(RESET)$(ORANGE) Cleanup $(RESET)$(DIM)──────────────────────────────────╮$(RESET)\n"
	@printf "$(DIM)│$(RESET)  $(BOLD)clean$(RESET)        Remove build artifacts       $(DIM)│$(RESET)\n"
	@printf "$(DIM)╰────────────────────────────────────────────╯$(RESET)\n"

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
