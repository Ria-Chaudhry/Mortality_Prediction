# Clinical Domain Mortality Framework

Dataset-agnostic framework for comparing the standalone and incremental prognostic value of early clinical domains for acute-care mortality prediction.

This repository is structured to reproduce the CHoRUS analysis, support MIMIC-IV replication, and make it practical for new OMOP or non-OMOP EHR sites to run the same comparative clinical-domain workflow without changing analytical core code.

## What This Is

This is not a single deployable mortality model. It is a reusable analysis framework that separates:

- dataset-specific extraction and harmonization;
- standardized intermediate schemas;
- feature construction by clinical domain;
- fold-safe preprocessing, feature selection, and model fitting;
- grouped validation and paired evaluation;
- manuscript and supplement output generation.

## Repository Layers

- `src/clinical_domains/core`: cohort logic, landmarks, outcomes, eligibility, audit, and validation.
- `src/clinical_domains/adapters`: adapter interfaces and starter implementations.
- `src/clinical_domains/features`: baseline, physiological, treatment, procedure, and matrix builders.
- `src/clinical_domains/modeling`: algorithms, hyperparameters, grouped cross-validation, model selection, and prediction.
- `src/clinical_domains/evaluation`: discrimination, calibration, threshold metrics, decision curves, top-risk analyses, and paired differences.
- `src/clinical_domains/reporting`: tables, figures, manuscript outputs, and supplement outputs.
- `configs`: reusable and dataset-specific configuration.
- `schemas` and `metadata`: standardized contracts and dictionaries.
- `examples/synthetic`: small safe data for end-to-end tests.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run the synthetic example:

```bash
python scripts/reproduce_manuscript.py --config examples/synthetic/config.yaml --output-dir outputs/synthetic_run
```

## Add a New Dataset

1. Start from `configs/generic_ehr/generic_ehr.example.yaml` or `configs/generic_omop/generic_omop.example.yaml`.
2. Implement an adapter that produces the standardized schemas documented in `schemas/`.
3. Keep SQL, item IDs, local code systems, and source-specific joins inside `adapters/`.
4. Run `python scripts/validate_config.py --config <your-config.yaml>`.
5. Run the pipeline with the same core scripts used for synthetic data.

## Data Governance

Patient-level restricted data must never be committed. Use local ignored paths such as `data/`, `restricted/`, or externally managed database connections. The committed example data in this repository is synthetic only.

## Git Setup

This folder is initialized as its own git repository. To push it:

```bash
git remote add origin <your-repo-url>
git branch -M main
git push -u origin main
```
