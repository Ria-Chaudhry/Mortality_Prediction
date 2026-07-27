# Clinical-domain mortality prediction framework

This repository implements the study design in
[`docs/pipeline_specification.pdf`](docs/pipeline_specification.pdf): predict death after a
24-hour admission landmark and within 30 days of acute-care admission from baseline factors and
first-24-hour measurement, medication, and procedure domains.

CHoRUS is the primary development analysis. MIMIC-IV is an independently normalized and trained
replication analysis. The datasets are never pooled: each has its own cohort, patients, folds,
concept selections, preprocessing, models, predictions, and manifests. MIMIC-IV is therefore a
replication of the domain-ranking pattern, not validation of a CHoRUS-trained model.

## What the pipeline runs

One source-neutral engine applies adult, non-elective acute-care eligibility; creates the 24-hour
landmark and verified 30-day outcome; derives prior-only, non-age-adjusted Charlson features;
assigns patients to one deterministic five-fold partition; and prepares qualifying first-24-hour
events.

Inside each outer fold, the other four folds alone choose 50 concepts per domain and fit unit
definitions, median imputation, one-hot encoding, variance filtering, and scaling. The selected
definitions are applied to the held-out fold without refitting. Measurements produce 300 features,
medications 104, and procedures 103. Eight matrices are evaluated with logistic regression,
random forest, gradient boosting, and LightGBM: 160 fits per dataset and exactly 32 OOF
probabilities per visit.

The pipeline saves numerical tables and plot-ready ROC, calibration, and decision-curve
coordinates. It generates no plots.

## Clean-clone installation

Use Python 3.10-3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make lint
make test
```

`requirements.lock` contains the same fully pinned versions installed by CI. The lock was tested,
not synthesized as documentation.

## Public synthetic demonstration

No credential, database, protected resource, or network access is needed after dependencies are
installed:

```bash
make synthetic-run
make verify
```

This regenerates committed CHoRUS-like and MIMIC-like inputs, runs both adapters independently, and
verifies aggregate schemas, design counts, expected summaries, and SHA-256 checksums. Source
identifiers and OOF rows are written only under ignored `restricted_outputs/`.

## CHoRUS execution

CHoRUS analysis requires authorized CHoRUS access and a site-confirmed mapping. Copy
`configs/chorus.example.yaml`, replace the synthetic table/column mappings with the confirmed
site mapping, change `backend` to `sql` when appropriate, and set only the named environment
variable:

```bash
export CHORUS_DATABASE_URL='provided-outside-version-control'
clinical-domain-mortality validate --config configs/chorus.site.yaml
clinical-domain-mortality run --dataset chorus --config configs/chorus.site.yaml
```

The adapter supports configurable OMOP-compatible person, visit, death, condition, measurement,
drug exposure, procedure occurrence, observation, and bridge structures. A mapping may use direct
visit, approved bridge, or patient-time linkage. Medication and procedure semantics remain the
source semantics; an order is not silently converted to an administration or performed procedure.

## MIMIC-IV execution

MIMIC-IV results require credentialed access to MIMIC-IV. This repository contains no MIMIC data.
Record the exact credentialed release version in a site config; the PDF does not identify a
release number, so this repository does not invent one. Point `source.root` to a user-supplied
local root containing the configured CSV, compressed CSV, or Parquet tables:

```bash
clinical-domain-mortality validate --config configs/mimic.site.yaml
clinical-domain-mortality run --dataset mimic --config configs/mimic.site.yaml
```

The supplied mapping documents admissions, patients, death records, ICD diagnoses, laboratory
events, prescriptions, and coded procedures. Alternative administration or performed-event
sources require an explicit table mapping and matching semantics. Deterministic subsampling is
configurable and occurs only inside the MIMIC run.

## Configuration workflow

Version-controlled scientific decisions live in:

- `configs/cohort.yaml`: visit types, age, lookback, window, landmark, horizon, folds, seeds, and
  restricted/public roots.
- `configs/features.yaml`: selection count/ranking, units, semantics, feature definitions, expected
  counts, and forbidden predictors.
- `configs/models.yaml`: eight matrices, four estimators, frozen hyperparameters, threads, and
  model order.
- `configs/evaluation.yaml`: thresholds, calibration bins, top-risk fraction, 2,000 patient
  bootstraps, decision thresholds, paired comparisons, and selection hierarchy.
- Dataset configs: physical tables/columns, semantics, access method, optional paper expected
  counts, and dataset version.

Paper runs may set the PDF counts as validation targets (CHoRUS: 22,098 visits, 5,892 patients,
1,004 deaths; MIMIC-IV: 23,000 visits, 10,009 patients, 819 deaths). They are not embedded in
reusable cohort logic. A mismatch hard-fails and is reported honestly.

## Stage order and commands

The numbered scripts call the same package functions as the CLI:

1. Validate source mapping and standardized schemas.
2. Freeze the cohort, outcome, baseline, row order, and hashes.
3. Create one patient-level five-fold assignment.
4. Validate clinical-domain mappings and explicit semantics.
5. Prepare qualifying first-24-hour events without global concept selection.
6. Select concepts and construct features independently per training fold.
7. Run all 160 outer-fold fits and validate OOF coverage.
8. Build all public-safe aggregate analyses and manifests.

Use `clinical-domain-mortality stage --stage N --config ...` or the corresponding script. Each
stage reruns its prerequisites so discovery never silently becomes analysis.

## Output boundary

Public run directories contain attrition, fold summaries, selection audits where disclosure is
allowed, pooled/fold performance, bootstrap intervals, paired comparisons, selected-model tables,
ROC/calibration/decision-curve coordinates, and dataset/fold/domain/matrix/model/run manifests.
See [`docs/output_dictionary.md`](docs/output_dictionary.md).

Raw clinical data, dates, identifiers, fold assignments, prepared event rows, fitted artifacts,
and patient-level OOF predictions are restricted. They are never required in a Git commit and are
ignored by default. The framework cannot reproduce CHoRUS findings without authorized CHoRUS data,
or MIMIC findings without the credentialed MIMIC release and confirmed configuration.

## Architecture and methods traceability

Adapters implement one field-level contract; common cohort, feature, modeling, evaluation, and
audit modules consume only standardized tables. Read
[`docs/source_adapter_contract.md`](docs/source_adapter_contract.md),
[`docs/architecture.md`](docs/architecture.md), and
[`docs/manuscript_methods_crosswalk.md`](docs/manuscript_methods_crosswalk.md).

Reproducible code does not establish transportability, causality, or deployment readiness. The
pooled-OOF 90%-specificity point is descriptive, not a prospectively validated operating
threshold.

## Citation and reuse

Citation metadata is in `CITATION.cff`. Only authorship confirmed by the finalized specification
is included.

No license is granted for reuse or redistribution.
