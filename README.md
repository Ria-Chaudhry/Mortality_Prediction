# Clinical-domain mortality prediction framework

This repository implements a leakage-controlled mortality-prediction workflow for a CHoRUS
primary analysis and an independently normalized and trained MIMIC-IV replication. It predicts
death after a 24-hour landmark and within 30 days of acute-care admission from baseline factors
and early measurement, medication, and procedure records.

The public synthetic execution is tested. Native MIMIC-IV file normalization is tested end to
end with native-shaped synthetic tables. CHoRUS column-projected SQL construction is regression
tested, but no protected CHoRUS database is available here. No real-data paper reproduction or
release-cleared clinical result is included or claimed.

## Study workflow

Each dataset is run independently; outcomes, patients, concepts, folds, models, and predictions
are never transferred or pooled. The pipeline:

1. projects configured columns and applies cohort/time predicates before loading large domains;
2. freezes adult, non-elective acute encounters, row order, the outcome, and baseline;
3. assigns patients, not encounters, to one deterministic five-fold partition;
4. within each outer fold and domain, ranks 50 concepts using distinct training visits only;
5. constructs 300 measurement, 104 medication, or 103 procedure candidate columns;
6. before imputation, ranks candidate columns by training-visit support proportion and retains
   exactly 21 final matrix columns per domain, resolving equal support by the frozen
   candidate-construction order;
7. reuses the same fold-specific 21 columns in every matrix containing that domain;
8. fits all learned preprocessing and each model on outer-training visits only, then creates one
   held-out positive-class probability per visit, matrix, and model.

The eight matrices are baseline, three single-domain additions, three pairwise additions, and all
domains. Logistic regression, random forest, gradient boosting, and LightGBM give 160 outer-fold
fits and 32 OOF probabilities per visit in each dataset.

The evidence search found a conflict: completed MIMIC stage scripts used training-fold mutual
information after median imputation, retained 15 measurement features and 21
medication/procedure features, and ranked 250 medication concepts; maintained manuscript
material describes occurrence frequency, 50 concepts, and 21 features in every domain. The
implemented synthetic rule is the requested unsupervised 21-column design, but it is not
represented as a confirmed historical paper method. Paper mode fails closed until the discrepancy
is reconciled. See
[`docs/recovered_method_provenance.md`](docs/recovered_method_provenance.md).

## Install and test

Frozen verification supports CPython 3.10.13 exactly:

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

The MIMIC synthetic run directly consumes official-shaped native tables. Verification pins the
complete deterministic aggregate artifact set, safe run-manifest fields, schemas, design counts,
and hashes. Patient-level cohort, feature, fold, event, and OOF files remain under ignored
`restricted_outputs/`.

Updating expected synthetic outputs is intentionally separate from verification and requires
review plus:

```bash
make freeze-synthetic-expected
```

## CHoRUS execution boundary

CHoRUS requires authorized access, a confirmed snapshot identifier, confirmed table/column
mappings, and approved measurement-unit rules. SQL access is supplied only through named
environment variables. The adapter selects configured columns, builds a server-side temporary
acute-cohort relation with bounded bulk inserts, and joins large domains to that relation and
their per-encounter predictor windows. It never uses `SELECT *` or a cohort-sized SQL `IN (...)`
parameter list.

`configs/chorus.paper.yaml` contains the expected paper counts but deliberately fails closed
because the snapshot, site mappings, units, top-21 override, event counts, and release governance
have not been confirmed:

```bash
make paper-run-chorus
make verify-paper-chorus
```

Those commands must not be treated as successful until a controlled local override resolves
every named field and an actual run passes reconciliation.

## Native MIMIC-IV execution boundary

MIMIC-IV requires credentialed PhysioNet access. The native adapter reads projected columns from
CSV, CSV.GZ, or Parquet using bounded chunks or Parquet predicate pushdown. Its required tables
and fields are documented in
[`docs/source_adapter_contract.md`](docs/source_adapter_contract.md).

Age is `anchor_age + (admission year - anchor_year)`. A precise
`admissions.deathtime` has priority and is never overridden by a midnight-coerced `patients.dod`;
`dod` is retained as a date-only fallback, and source conflicts are audited. A date-only death on
the landmark calendar date is conservatively excluded without inventing a time. Medication
concepts must be configured as one of
`formulary_drug_cd`, `gsn`, `ndc`, or `drug`; race, ethnicity availability/derivation, and
admission types require explicit harmonization. Native records without event IDs receive stable, multiplicity-preserving internal
keys. Concepts are namespaced by source. `procedures_icd.chartdate` remains date-only and uses the
recovered inclusive calendar-date-span rule rather than a fictitious timestamp window.

The exact MIMIC-IV manuscript release could not be established from the PDF, repository history,
old scripts, or local documentation. `configs/mimiciv.paper.yaml` leaves it `UNCONFIRMED` and
fails closed; it does not assume v3.1 or “current”:

```bash
export MIMICIV_ROOT=/authorized/local/mimic/root
make paper-run-mimiciv
make verify-paper-mimiciv
```

## Predictor window, outcome, and Charlson

`predictor_window_hours` controls extraction:
`[admission, min(discharge when known, admission + predictor window))`.
`landmark_hours` independently controls early-death exclusion and prediction time. Configuration
fails if the predictor window exceeds the landmark without a documented override.

The non-age Charlson score uses only diagnoses on prior acute admissions starting in the
configured 365-day lookback and excludes the index admission. ICD-9-CM and ICD-10-CM are validated
and classified separately using the Quan/Deyo algorithm, with diabetes, liver, and malignancy
hierarchies. See [`docs/charlson.md`](docs/charlson.md).

## Outputs and privacy

Artifact states are `restricted`, `release_candidate_aggregate`, `public_clinical`, and
`public_synthetic`. Real artifacts default to `restricted`; a real run does not write
release-candidate tables to the public output root. Public clinical release requires an explicit
allowlisted schema, governance-approved small-cell threshold, release approval, and a recorded
successful `public_clinical` scan. An approval flag alone is insufficient. Unit audits are public
only for synthetic runs by default.

Analytical inputs are hashed from canonical, cohort-restricted values, not only schemas and row
counts. Manifests distinguish `feature_schema_hash` from `feature_value_hash` or
`feature_matrix_hash`. The latter changes when any row identity, column order, or feature value
changes without exposing those values.

See [`SECURITY_AND_PRIVACY.md`](SECURITY_AND_PRIVACY.md) and
[`docs/output_dictionary.md`](docs/output_dictionary.md).

## Reproducibility status

The synthetic implementation, both adapters, leakage barriers, deterministic selection, and
aggregate calculations are testable from a clean clone. Exact CHoRUS and MIMIC manuscript results
require the unavailable authorized snapshots and confirmed local decisions. A count mismatch
writes attrition, expected-versus-observed counts, and a failed diagnostic manifest before
stopping. Paper verification also requires manuscript reconciliation and release clearance.

Selected models additionally receive held-out permutation-SHAP analysis using an outer-training
background. Only fold-level mean absolute SHAP and cross-fold aggregate tables are written; the
historical unified eight-matrix SHAP method remains unreconciled.

Methods-to-code traceability is in
[`docs/manuscript_methods_crosswalk.md`](docs/manuscript_methods_crosswalk.md). Citation metadata
is in `CITATION.cff`.

No license is granted for reuse or redistribution.
