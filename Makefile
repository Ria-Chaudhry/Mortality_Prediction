PYTHON ?= python3
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib

.PHONY: install lint test test-full synthetic-run verify freeze-synthetic-expected paper-run-chorus paper-run-mimiciv verify-paper-chorus verify-paper-mimiciv clean

install:
	$(PYTHON) -m pip install --disable-pip-version-check -r requirements.lock
	$(PYTHON) -m pip install --disable-pip-version-check --no-deps -e .

lint:
	$(PYTHON) -m ruff check src tests scripts synthetic_data/generate_synthetic_data.py

test:
	$(PYTHON) -m pytest -m "not slow"

test-full:
	$(PYTHON) -m pytest

synthetic-run:
	$(PYTHON) synthetic_data/generate_synthetic_data.py
	$(PYTHON) -m clinical_domain_mortality synthetic-run

verify:
	$(PYTHON) -m clinical_domain_mortality verify --run-dir outputs/synthetic

freeze-synthetic-expected:
	$(PYTHON) -m clinical_domain_mortality freeze-synthetic-expected --run-dir outputs/synthetic --approve-update

paper-run-chorus:
	$(PYTHON) -m clinical_domain_mortality run --dataset chorus --config configs/chorus.paper.yaml

paper-run-mimiciv:
	$(PYTHON) -m clinical_domain_mortality run --dataset mimiciv --config configs/mimiciv.paper.yaml

verify-paper-chorus:
	$(PYTHON) -m clinical_domain_mortality verify-paper --config configs/chorus.paper.yaml --run-dir restricted_outputs/paper/chorus/release_candidate_aggregate

verify-paper-mimiciv:
	$(PYTHON) -m clinical_domain_mortality verify-paper --config configs/mimiciv.paper.yaml --run-dir restricted_outputs/paper/mimiciv/release_candidate_aggregate

clean:
	$(PYTHON) -c "from pathlib import Path; import shutil; [shutil.rmtree(p, ignore_errors=True) for p in [Path('outputs/synthetic'), Path('restricted_outputs/synthetic')]]"
