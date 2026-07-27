PYTHON ?= python3
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib

.PHONY: install lint test synthetic-run verify clean

install:
	$(PYTHON) -m pip install --disable-pip-version-check -r requirements.lock
	$(PYTHON) -m pip install --disable-pip-version-check --no-deps -e .

lint:
	$(PYTHON) -m ruff check src tests scripts synthetic_data/generate_synthetic_data.py

test:
	$(PYTHON) -m pytest -m "not slow"

synthetic-run:
	$(PYTHON) synthetic_data/generate_synthetic_data.py
	$(PYTHON) -m clinical_domain_mortality synthetic-run

verify:
	$(PYTHON) -m clinical_domain_mortality verify --run-dir outputs/synthetic

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('outputs/synthetic'), Path('restricted_outputs/synthetic')]]"
