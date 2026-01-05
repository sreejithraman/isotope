# Contributing to Isotope

## Development Setup

```bash
# Clone and setup (installs deps + pre-commit hooks)
git clone <repo>
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

## Before Submitting a PR

Pre-commit hooks run automatically on commit. If you need to fix issues manually:

```bash
make fix
```
