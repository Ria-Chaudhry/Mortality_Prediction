PYTHON ?= python3
export MPLCONFIGDIR ?= $(CURDIR)/.cache/matplotlib
export PYTHONHASHSEED ?= 0
export PYTHONFAULTHANDLER ?= 1
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export NUMEXPR_NUM_THREADS ?= 1
export VECLIB_MAXIMUM_THREADS ?= 1
export NUMBA_NUM_THREADS ?= 1

.PHONY: install lint test test-full synthetic-run verify freeze-synthetic-expected paper-preflight-chorus paper-preflight-mimiciv paper-run-chorus paper-run-mimiciv verify-paper-chorus verify-paper-mimiciv clean

install:
	$(PYTHON) -m pip install --disable-pip-version-check -r requirements.lock
	$(PYTHON) -m pip install --disable-pip-version-check --no-deps -e .

lint:
	$(PYTHON) -m ruff check src tests scripts synthetic_data/generate_synthetic_data.py

test:
	$(PYTHON) -m pytest -m "not slow"

test-full:
	mkdir -p outputs/test-reports
	$(PYTHON) -X faulthandler -m pytest --junitxml=outputs/test-reports/pytest-full.xml

synthetic-run:
	$(PYTHON) synthetic_data/generate_synthetic_data.py
	$(PYTHON) -m clinical_domain_mortality synthetic-run

verify:
	$(PYTHON) -m clinical_domain_mortality verify --run-dir outputs/synthetic

freeze-synthetic-expected:
	$(PYTHON) -m clinical_domain_mortality freeze-synthetic-expected --run-dir outputs/synthetic --approve-update

paper-preflight-chorus:
	$(PYTHON) -m clinical_domain_mortality paper-preflight --config configs/chorus.paper.yaml

paper-preflight-mimiciv:
	$(PYTHON) -m clinical_domain_mortality paper-preflight --config configs/mimiciv.paper.yaml

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
