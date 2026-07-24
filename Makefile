.RECIPEPREFIX := >

.PHONY: install test lint format synthetic docs clean

install:
> python -m pip install -e ".[dev]"

test:
> pytest

lint:
> ruff check src scripts tests

format:
> black src scripts tests

synthetic:
> python scripts/reproduce_manuscript.py --config examples/synthetic/config.yaml --output-dir outputs/synthetic_run

docs:
> mkdocs serve

clean:
> find . -type d -name "__pycache__" -prune -exec rm -rf {} +
> find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
