# Clinical-domain mortality prediction framework

This repository implements a mortality-prediction workflow for a CHoRUS primary analysis and an independent MIMIC-IV replication. It predicts death after a 24-hour landmark and within 30 days of acute-care admission from baseline factors and early measurement, medication, and procedure records.

The public synthetic execution is tested. Native MIMIC-IV file normalization is tested end to end with native-shaped synthetic tables. CHoRUS column-projected cohort-first SQL planning is regression tested, but no protected CHoRUS database has been accessed. No real-data paper reproduction or release-cleared clinical result is included or claimed.

## Study workflow

Each dataset is run independently; outcomes, patients, concepts, folds, models, and predictions are never transferred or pooled. The pipeline:

1. Projects configured columns and applies cohort/time predicates before loading large domains.
2. Freezes adult, non-elective acute encounters, row order, the outcome, and baseline.
3. Assigns patients, not encounters, to one deterministic five-fold partition.
4. Within each outer fold and domain, ranks 50 concepts using distinct training visits only.
5. Constructs 300 measurement, 104 medication, or 103 procedure candidate columns.
6. Before imputation, ranks candidate columns by training-visit support proportion and retains exactly 21 final matrix columns per domain, resolving equal support by the frozen candidate-construction order.
7. Reuses the same fold-specific 21 columns in every matrix containing that domain.
8. Fits all learned preprocessing and each model on outer-training visits only, then creates one held-out positive-class probability per visit, matrix, and model.

The eight matrices are baseline, three single-domain additions, three pairwise additions, and all domains. Logistic regression, random forest, gradient boosting, and LightGBM give 160 outer-fold fits and 32 OOF probabilities per visit in each dataset.

The evidence search found a conflict: completed MIMIC stage scripts used training-fold mutual information after median imputation, retained 15 measurement features and 21 medication/procedure features, and ranked 250 medication concepts; maintained manuscript material describes occurrence frequency, 50 concepts, and 21 features in every domain. The implemented synthetic rule is the requested unsupervised 21-column design, but it is not represented as a confirmed historical paper method. Paper mode fails closed until the discrepancy is reconciled. See [`docs/recovered_method_provenance.md`](docs/recovered_method_provenance.md).

## Install and test

Execution and same-runtime repeat checks support CPython 3.10.13 exactly:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make lint
make test
make test-full
```

`requirements.lock` is fully pinned and is the file installed by CI.

## Public synthetic demonstration

No credential, protected data, database, or network resource is needed after installation:

```bash
make synthetic-run
make verify
```

The MIMIC synthetic run directly consumes official-shaped native tables. Verification pins the complete deterministic aggregate artifact set, safe run-manifest fields, schemas, design counts, and canonical calculation hashes. Public floating-point outputs are serialized at ten decimal places. Derived floating-point features are first canonicalized at the configured eight-decimal boundary before hashing, preprocessing, or fitting so platform-level aggregation noise cannot change tree split ties. Same-run manifests verify exact file bytes.

Exact frozen fitted-model verification is intentionally narrower because pinned scikit-learn tree builds can make different tied split choices across operating systems even with identical inputs, versions, seeds, and one thread. The reference platform is CPython 3.10.13 on Linux x86_64; CI pins Ubuntu 24.04. `make verify` fails closed elsewhere rather than accepting a different fitted-model baseline. On another platform, dataset-level structural and exact same-run checks remain available with:

```bash
clinical-domain-mortality verify --run-dir outputs/synthetic/chorus
clinical-domain-mortality verify --run-dir outputs/synthetic/mimiciv
```

Patient-level cohort, feature, fold, event, and OOF files remain under ignored `restricted_outputs/`.

Updating expected synthetic outputs is intentionally separate from verification and requires review plus:

```bash
make freeze-synthetic-expected
```

## CHoRUS execution boundary

CHoRUS requires authorized access, a confirmed snapshot identifier, confirmed table/column mappings, and approved measurement-unit rules. SQL access is supplied only through named environment variables. The adapter selects configured columns, builds a server-side temporary eligible acute-cohort relation directly from mapped source tables, and joins large domains to that relation and their per-encounter predictor windows. It never uses `SELECT *` or a cohort-sized SQL `IN (...)` parameter list.

`configs/chorus.paper.yaml` contains the expected paper counts but deliberately fails closed because the snapshot, site mappings, units, top-21 override, event counts, and release governance have not been confirmed:

```bash
make paper-preflight-chorus
make paper-run-chorus
make verify-paper-chorus
```

Those commands must not be treated as successful until a controlled local override resolves every named field and an actual run passes reconciliation.

## Native MIMIC-IV execution boundary

MIMIC-IV requires credentialed PhysioNet access. The native adapter reads projected columns from CSV, CSV.GZ, or Parquet using bounded chunks or Parquet predicate pushdown. Its required tables and fields are documented in [`docs/source_adapter_contract.md`](docs/source_adapter_contract.md).

Age is `anchor_age + (admission year - anchor_year)`. A precise `admissions.deathtime` has priority and is never overridden by a midnight-coerced `patients.dod`; `dod` is retained as a date-only fallback, and source conflicts are audited. A date-only death on the landmark calendar date is conservatively excluded without inventing a time. Medication concepts must be configured as one of `formulary_drug_cd`, `gsn`, `ndc`, or `drug`; race, ethnicity availability/derivation, and admission types require explicit harmonization. Native records without event IDs receive stable, multiplicity-preserving internal keys. Concepts are namespaced by source. `procedures_icd.chartdate` remains date-only and uses the recovered inclusive calendar-date-span rule rather than a fictitious timestamp window.

The exact MIMIC-IV manuscript release could not be established from the PDF, repository history, old scripts, or local documentation. `configs/mimiciv.paper.yaml` leaves it `UNCONFIRMED` and fails closed; it does not assume v3.1 or “current”:

```bash
make paper-preflight-mimiciv
make paper-run-mimiciv
make verify-paper-mimiciv
```

The preflight commands inspect configuration only and never open a database or clinical source. Authorized local deployments supply private roots and credentials through environment variables or an ignored override only after every reported blocker is resolved.

## Predictor window, outcome, and Charlson

`predictor_window_hours` controls extraction: `[admission, min(discharge when known, admission + predictor window))`.

`landmark_hours` independently controls early-death exclusion and prediction time. Configuration fails if the predictor window exceeds the landmark without a documented override.

The non-age Charlson score uses only diagnoses on prior acute admissions starting in the configured 365-day lookback and excludes the index admission. ICD-9-CM and ICD-10-CM are validated and classified separately using the Quan/Deyo algorithm, with diabetes, liver, and malignancy hierarchies. See [`docs/charlson.md`](docs/charlson.md).

## Outputs and privacy

Artifact states are `restricted`, `release_candidate_aggregate`, `public_clinical`, and `public_synthetic`. Real artifacts default to `restricted`; a real run does not write release-candidate tables to the public output root. Public clinical release requires an explicit allowlisted schema, governance-approved small-cell threshold, release approval, and a recorded successful `public_clinical` scan. An approval flag alone is insufficient. Unit audits are public only for synthetic runs by default.

Analytical inputs are hashed from canonical, cohort-restricted values, not only schemas and row counts. Fit manifests also hash the exact frozen training, validation, and preprocessing-fit partitions and the fitted imputation/encoding/variance/scaling state. Manifests distinguish `feature_schema_hash` from `feature_value_hash` or `feature_matrix_hash`. The latter changes when any row identity, column order, or feature value changes without exposing those values.

See [`SECURITY_AND_PRIVACY.md`](SECURITY_AND_PRIVACY.md) and [`docs/output_dictionary.md`](docs/output_dictionary.md).

## Reproducibility status

The synthetic implementation, both adapters, leakage barriers, deterministic selection, and aggregate calculations are testable from a clean clone. Exact CHoRUS and MIMIC manuscript results require the unavailable authorized snapshots and confirmed local decisions. A count mismatch writes attrition plus final, attrition-stage, event-stage, or fold/domain selection comparisons and a failed diagnostic manifest before stopping. Paper verification recomputes the actual top-50/top-21 evidence, OOF fold identity, and matrix hashes; it also requires manuscript reconciliation and a completed `public_clinical` release gate.

Selected models additionally receive held-out permutation-SHAP analysis using an outer-training background. Only fold-level mean absolute SHAP and cross-fold aggregate tables are written; the historical unified eight-matrix SHAP method remains unreconciled.

Methods-to-code traceability is in [`docs/manuscript_methods_crosswalk.md`](docs/manuscript_methods_crosswalk.md). Citation metadata is in `CITATION.cff`.

No license is granted for reuse or redistribution.

## What this repository does

The repository provides an executable, end-to-end framework for reproducing the study once authorized source data and confirmed study configurations are supplied. It:

- Constructs landmarked acute-care mortality cohorts independently in CHoRUS and MIMIC-IV.
- Derives baseline, physiological-severity, treatment-exposure, and procedure-burden predictors.
- Enforces patient-level separation and training-fold-only feature selection and preprocessing.
- Evaluates eight prespecified feature matrices using logistic regression, random forest, gradient boosting, and LightGBM.
- Generates held-out out-of-fold predictions and aggregate discrimination, calibration, clinical-utility, decision-curve, and feature-importance outputs.
- Records cohort attrition, source mappings, feature-selection decisions, partitions, configuration values, and reproducibility hashes.
- Prevents patient-level data and unapproved clinical outputs from entering public artifacts.
- Provides a complete privacy-safe synthetic demonstration for testing the implemented workflow.
- Provides fail-closed paper-mode entry points for authorized CHoRUS and MIMIC-IV reproduction.

The repository contains executable implementation code rather than pseudocode. However, the CHoRUS-specific implementation has not been run or validated against the protected CHoRUS environment, so it should currently be described as an unvalidated source-specific implementation rather than a completed reproduction of the CHoRUS analysis.

## What is left

The following work is required before the repository can be described as reproducing the manuscript analyses:

1. Reconcile the feature-selection discrepancy between the completed historical scripts and the maintained manuscript description, including the ranking method, number of candidate medication concepts, and number of retained features per domain.
2. Confirm the final SHAP procedure and reconcile it with the feature-importance results reported in the manuscript.
3. Confirm the exact CHoRUS snapshot, source tables, column mappings, encounter definitions, concept mappings, measurement-unit rules, and governance requirements.
4. Confirm the exact MIMIC-IV release and the final race, ethnicity, admission-type, medication, procedure, and death-date mappings.
5. Tighten the frozen-runtime check so that it explicitly distinguishes the Ubuntu 24.04 reference environment from other Linux x86_64 systems.
6. Run the finalized pipeline on authorized CHoRUS and MIMIC-IV data.
7. Reconcile cohort attrition, final cohort counts, event counts, fold assignments, selected concepts and features, OOF predictions, performance estimates, clinical-utility analyses, calibration results, decision curves, and SHAP summaries with the manuscript.
8. Complete the required disclosure and privacy review before releasing any aggregate clinical output.
9. Obtain supervisor feedback on the final methodological specification, the framing of the CHoRUS code, and whether an authorized CHoRUS analyst can validate the mappings and execute the pipeline inside the private environment.
10. Decide whether to add an open-source license before permitting reuse or redistribution.


