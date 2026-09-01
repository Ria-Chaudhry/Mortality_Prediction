# Clinical-Domain Mortality Prediction Framework

This repository contains the analysis framework and implementation associated with:

**Assessing Clinical Features for Mortality Prediction in CHoRUS Clinical Care for AI and MIMIC-IV Datasets**

The study evaluates the relative predictive value of routinely available early clinical data streams for visit-level 30-day mortality prediction using:

- CHoRUS Clinical Care for AI version 1
- MIMIC-IV version 3.1

The repository provides code for cohort construction, feature construction, patient-grouped cross-validation, model evaluation, and secondary analyses. Patient-level CHoRUS and MIMIC-IV data are not included.

## Study objective

The analysis evaluates four clinical data streams:

1. Baseline vulnerability
2. Physiological severity
3. Treatment exposure
4. Procedure burden

The goal is to quantify the incremental and complementary predictive information contributed by these domains within a common modeling framework.

Eight feature matrices are evaluated:

- Baseline
- Baseline + physiological severity
- Baseline + treatment exposure
- Baseline + procedure burden
- Baseline + physiological severity + treatment exposure
- Baseline + physiological severity + procedure burden
- Baseline + treatment exposure + procedure burden
- Baseline + all clinical data streams

CHoRUS and MIMIC-IV are analyzed independently. Models, patients, predictions, and outcome labels are not transferred between datasets.

## Prediction window and outcome

The prediction anchor is 24 hours after encounter start.

For encounters lasting at least 24 hours, predictors are restricted to the first 24 hours after encounter start.

For encounters lasting less than 24 hours, the entire encounter from encounter start through encounter end is used as the predictor window.

Encounters in which death occurred before or at 24 hours after encounter start are excluded because the outcome had already occurred by the prediction landmark.

The primary outcome is:

> Death occurring after 24 hours and within 30 days of encounter start.

Discharge-related variables are not used as predictors.

## Cohorts

### CHoRUS

The final CHoRUS analytic cohort contains:

- 22,098 acute-care visits
- 5,892 unique patients
- 1,004 30-day mortality events
- 4.5% mortality prevalence

Eligible acute-care encounters include hospital, observation, inpatient, emergency, and combined emergency/inpatient visit types.

### MIMIC-IV

The MIMIC-IV replication cohort contains:

- 23,000 acute-care visits
- 10,006 unique patients
- 819 30-day mortality events
- 3.6% mortality prevalence

MIMIC-IV version 3.1 is used.

A reproducible patient-level subsampling procedure preserves complete patient clusters so that encounters from the same patient are not independently sampled across validation folds.

Across both datasets, the study includes:

- 45,098 visits
- 15,898 unique patients

## Baseline features

Baseline vulnerability represents information available before or at the beginning of the acute-care encounter.

Predictors include:

- Age at visit
- Sex
- Race
- Ethnicity
- Visit type
- Prior visit count
- Prior acute-care visit count
- Indicator for prior utilization
- Charlson Comorbidity Index

Categorical variables are one-hot encoded for modeling.

The CHoRUS baseline model matrix contains 21 model-ready features.

The MIMIC-IV baseline data contain nine raw baseline predictors that yield 25 encoded model features after categorical encoding.

## Clinical-domain feature construction

### Physiological severity

Physiological features are derived from measurements recorded during the predictor window.

For selected measurement concepts, visit-level summaries include:

- Mean
- Minimum
- Maximum
- Standard deviation
- Count
- Missingness indicator

Candidate physiological features are generated from frequently occurring measurement concepts.

### Treatment exposure

Treatment-exposure features are derived from medication records during the predictor window.

Features include:

- Medication exposure indicators
- Medication exposure counts
- Number of unique medications
- Repeated medication exposure count
- Time to first medication
- Any-medication indicator

### Procedure burden

Procedure-burden features are derived from procedures recorded during the predictor window.

Features include:

- Procedure presence indicators
- Procedure counts
- Unique procedure count
- Total procedure count
- Any-procedure indicator

## Feature selection

Feature selection is performed independently within each cross-validation training fold.

For each clinical domain, candidate features are ranked according to their occurrence frequency within the training partition.

The 21 most frequently occurring domain features are retained and applied unchanged to the corresponding held-out validation partition.

Selection is therefore performed without using information from the held-out fold.

The same fold-specific feature set is reused across every feature matrix containing that clinical domain.

This procedure standardizes the number of added features across clinical domains and limits information leakage.

## Cross-validation

Five-fold patient-level grouped cross-validation is used.

All encounters belonging to the same patient remain in the same fold. This prevents encounters from one patient from appearing in both the training and validation partitions of the same cross-validation iteration.

All preprocessing and feature-selection operations that depend on the data are fit using training-fold data only and then applied to the held-out fold.

## Models

Four machine-learning algorithms are evaluated for each feature matrix:

- Logistic regression
- Random forest
- Gradient boosting
- Light Gradient Boosting Machine (LightGBM)

The primary analysis selects the algorithm with the highest mean cross-validated area under the precision-recall curve (AUPRC) for each feature matrix.

Reported primary performance is summarized across held-out folds.

A fixed-algorithm sensitivity analysis is also performed to evaluate whether the clinical-domain findings depend on matrix-specific algorithm selection.

## Evaluation

Primary model-performance measures include:

- Area under the precision-recall curve (AUPRC)
- Area under the receiver operating characteristic curve (AUROC)
- Brier score

Additional evaluation includes:

- Calibration
- Sensitivity at approximately 90% specificity
- Positive predictive value at that operating point
- Top-10% risk analysis
- Decision-curve analysis
- SHAP feature attribution
- Pairwise top-feature interaction analysis
- Age subgroup analysis
- Sex subgroup analysis

Scenario analyses additionally estimate the number of subsequently fatal encounters that would be identified under prespecified hypothetical improvements in sensitivity. These are scenario-based estimates and do not represent observed treatment effects or preventable deaths.

## Repository structure

```text
.
├── configs/                         # Dataset and analysis configuration
├── docs/                            # Documentation and implementation notes
├── examples/                        # Example workflows
├── mappings/                        # Source-data mappings
├── outputs/                         # Release-cleared/public outputs where applicable
├── scripts/                         # Supporting analysis scripts
├── src/clinical_domain_mortality/   # Core Python package
├── synthetic_data/                  # Public synthetic demonstration data
├── tests/                           # Automated tests
├── run_pipeline.py                  # Pipeline entry point
├── requirements.lock                # Pinned Python dependencies
├── CITATION.cff
├── DATA_AVAILABILITY.md
├── SECURITY_AND_PRIVACY.md
└── LICENSE
