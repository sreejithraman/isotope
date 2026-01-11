# Contributing to Isotope

## Development Setup

```bash
# Clone and setup (installs deps + pre-commit hooks)
git clone https://github.com/sreejithraman/isotope.git
cd isotope
make dev-setup
```

## Common Commands

```bash
make fix        # Auto-fix lint/format issues
make lint       # Check for issues (same as CI)
make test       # Run tests
make typecheck  # Run mypy
```

## Running the App

```bash
make cli                              # Run CLI (auto-installs if needed)
make cli ARGS="ingest examples/data"  # With arguments
make cli ARGS="query 'what is X'"
make tui                              # Run TUI (auto-installs if needed)
```

These commands auto-install dependencies and skip reinstall when `pyproject.toml` hasn't changed.

## Running Tests

Tests require all dependencies:

```bash
pip install -e ".[dev,all]"
make test
```

This matches what CI runs.

## Before Submitting a PR

Pre-commit hooks run automatically on commit. If you need to fix issues manually:

```bash
make fix
```

## CI Structure

```
format ──┐
lint ────┼──► type-check ──┐
         │                 │
         ├──► test ────────┼──► ci (gate)
         │                 │
security─┘
```

The `ci` gate job aggregates all results into a single required check for branch protection.

**Adding new CI jobs:** Add the job to `.github/workflows/ci.yml`, then add it to the `needs` list in the `ci` gate job.

**Python 3.13:** Tested experimentally—failures don't block PRs until dependencies fully support it.
