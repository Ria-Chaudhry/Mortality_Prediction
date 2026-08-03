# Clinical-domain mortality prediction framework

This repository implements a mortality prediction workflow for a CHoRUS primary analysis and an independent MIMIC-IV replication. It predicts death after a 24 hour landmark and within 30 days of acute-care admission from baseline factors and early measurement, medication, and procedure records.

## Study workflow

Each dataset is run independently; outcomes, patients, concepts, folds, models, and predictions are never transferred or pooled. The pipeline:

1. Projects configured columns and applies cohort/time predicates before loading large domains.
2. Freezes the configured age range, non-elective acute encounters, row order, the outcome, and baseline.
3. Assigns patients, not encounters, to one frozen five-fold partition. MIMIC
   paper mode reproduces the completed patient/start/visit row order and seeded
   shuffled `StratifiedGroupKFold`; other runs use their configured versioned
   grouped-fold policy.
4. Within each outer fold and domain, ranks 50 concepts using distinct training visits only.
5. Constructs 300 measurement, 104 medication, or 103 procedure candidate columns.
6. Retains exactly 21 final matrix columns per domain using the configured outer-training-only rule: support prevalence for the synthetic demonstration and the recovered training median/mutual information rule for MIMIC paper mode.
7. Reuses the same fold specific 21 columns in every matrix containing that domain.
8. Fits all learned preprocessing and each model on outer-training visits only, then creates one held-out positive-class probability per visit, matrix, and model.

The eight matrices are baseline, three single domain additions, three pairwise additions, and all domains. Logistic regression, random forest, gradient boosting, and LightGBM give 160 outer fold fits and 32 OOF probabilities per visit in each dataset.

## Install and test

Execution and same runtime repeat checks support CPython 3.10.13 exactly:

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

The MIMIC synthetic run directly consumes official shaped native tables. Verification pins the complete deterministic aggregate artifact set, safe run manifest fields, schemas, design counts, and canonical calculation hashes. Public floating-point outputs are serialized at ten decimal places. Derived floating-point features are first canonicalized at the configured eight decimal boundary before hashing, preprocessing, or fitting so platform level aggregation noise cannot change tree split ties. Same run manifests verify exact file bytes.

Exact frozen fitted model verification is intentionally narrower because pinned scikit-learn tree builds can make different tied split choices across operating systems even with identical inputs, versions, seeds, and one thread. The reference platform is CPython 3.10.13 on Linux x86_64; CI pins Ubuntu 24.04. `make verify` fails closed elsewhere rather than accepting a different fitted model baseline. On another platform, dataset-level structural and exact same run checks remain available with:

```bash
clinical-domain-mortality verify --run-dir outputs/synthetic/chorus
clinical-domain-mortality verify --run-dir outputs/synthetic/mimiciv
```

Patient level cohort, feature, fold, event, and OOF files remain under ignored `restricted_outputs/`.

Updating expected synthetic outputs is intentionally separate from verification and requires review plus:

```bash
make freeze-synthetic-expected
```

## CHoRUS execution boundary

CHoRUS requires authorized access, a confirmed snapshot identifier, confirmed table/column mappings, and approved measurement-unit rules. SQL access is supplied only through named environment variables. The adapter selects configured columns, builds a server side temporary eligible acute cohort relation directly from mapped source tables, and joins large domains to that relation and their per-encounter predictor windows. It never uses `SELECT *` or a cohort-sized SQL `IN (...)` parameter list.

`configs/chorus.paper.yaml` contains the expected paper counts but deliberately fails closed because the snapshot, site mappings, units, top 21 override, event counts, and release governance have not been confirmed:

```bash
make paper-preflight-chorus
make paper-run-chorus
make verify-paper-chorus
```

Those commands must not be treated as successful until a controlled local override resolves every named field and an actual run passes reconciliation.

## Native MIMIC-IV execution boundary

MIMIC-IV requires credentialed PhysioNet access. The native adapter reads projected columns from CSV, CSV.GZ, or Parquet using bounded chunks or Parquet predicate pushdown. Its required tables and fields are documented in [`docs/source_adapter_contract.md`](docs/source_adapter_contract.md).

Age is `anchor_age + (admission year - anchor_year)`. The current MIMIC paper configuration uses minimum age 0, an explicit task-level change from the historical script’s age 18. Paper mortality reproduces the completed date-level rule: normalize `admissions.deathtime` and `patients.dod`, choose the earliest date, exclude on/before the landmark date, and label through day 30. The generic native adapter also supports a separate explicit precision-preserving rule for nonpaper analyses. Medication concepts use the recovered GSN → NDC → formulary code → normalized drug-name hierarchy. Race/ethnicity and the seven acute admission categories reproduce the completed mapping. Paper extraction requires a qualifying native `hadm_id` and uses admission through hour 24 without shortening the predictor interval at an early discharge, matching the completed scripts; patient-time fallback and discharge-capped windows remain explicit nonpaper options. Native records without event IDs receive stable, multiplicity-preserving internal keys. Concepts are namespaced by source. `procedures_icd.chartdate` remains date-only and uses the recovered inclusive calendar-date-span rule rather than a fictitious timestamp window.

The completed scripts identify MIMIC-IV v3.1. `configs/mimiciv.paper.yaml` freezes that release and all recovered source rules:

```bash
make paper-preflight-mimiciv
make paper-run-mimiciv
make verify-paper-mimiciv
```

The preflight command inspects configuration only and never opens a clinical source. Authorized execution supplies `MIMICIV_ROOT` and confirms `MIMICIV_RELEASE=v3.1`. Paper verification still requires actual count, selection, model, manuscript, and privacy reconciliation; configuration preflight alone is not a paper reproduction.

## Predictor window, outcome, and Charlson

`predictor_window_hours` controls extraction: `[admission, min(discharge when known, admission + predictor window))`.

`landmark_hours` independently controls early death exclusion and prediction time. Configuration fails if the predictor window exceeds the landmark without a documented override.

The non age Charlson score uses only diagnoses on prior acute admissions starting in the configured 365 day lookback and excludes the index admission. ICD-9-CM and ICD-10-CM are validated and classified separately using the Quan/Deyo algorithm, with diabetes, liver, and malignancy hierarchies. See [`docs/charlson.md`](docs/charlson.md).

## Outputs and privacy

Artifact states are `restricted`, `release_candidate_aggregate`, `public_clinical`, and `public_synthetic`. Real artifacts default to `restricted`; a real run does not write release-candidate tables to the public output root. Public clinical release requires an explicit allowlisted schema, governance-approved small-cell threshold, release approval, and a recorded successful `public_clinical` scan. An approval flag alone is insufficient. Unit audits are public only for synthetic runs by default.

Analytical inputs are hashed from canonical, cohort restricted values, not only schemas and row counts. Fit manifests also hash the exact frozen training, validation, and preprocessing fit partitions and the fitted imputation/encoding/variance/scaling state. Manifests distinguish `feature_schema_hash` from `feature_value_hash` or `feature_matrix_hash`. The latter changes when any row identity, column order, or feature value changes without exposing those values.

See [`SECURITY_AND_PRIVACY.md`](SECURITY_AND_PRIVACY.md) and [`docs/output_dictionary.md`](docs/output_dictionary.md).

## Reproducibility status

The synthetic implementation, both adapters, leakage barriers, deterministic selection, and aggregate calculations are testable from a clean clone. Exact CHoRUS and MIMIC manuscript results require authorized data and successful reconciliation. A count mismatch writes attrition plus final, attrition stage, event stage, or fold/domain selection comparisons and a failed diagnostic manifest before stopping. Paper verification recomputes the actual top-50/top-21 evidence, OOF fold identity, and matrix hashes; it also requires manuscript reconciliation and a completed `public_clinical` release gate.

One unified stage applies held out permutation SHAP to the selected model for every matrix and fold. It verifies a deterministic reconstruction of each selected fit against stored OOF probabilities, uses an outer training background and outer validation evaluation sample, preserves model feature order, and writes only fold-level mean absolute and cross-fold aggregate values.

Methods-to-code traceability is in [`docs/manuscript_methods_crosswalk.md`](docs/manuscript_methods_crosswalk.md). Citation metadata is in `CITATION.cff`.

## License

This repository is released under the MIT License. The license applies only to the code in this repository.

No patient-level CHoRUS or MIMIC-IV data are included or redistributed. Users are responsible for obtaining independent access to CHoRUS and/or MIMIC-IV and for complying with all applicable data-use agreements, institutional approvals, and governance requirements.
