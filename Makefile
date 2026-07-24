.PHONY: install test lint format synthetic verify docs clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src scripts tests

format:
	black src scripts tests

synthetic:
	python scripts/reproduce_manuscript.py --config examples/synthetic/config.yaml --output-dir outputs/synthetic_run

verify:
	python scripts/validate_environment.py
	pytest
	python scripts/verify_pipeline.py

docs:
	mkdocs serve

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +
