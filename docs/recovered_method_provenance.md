# Recovered analysis-method provenance

This document records method evidence found during the July 2026 repository
rebuild. It records only basenames, line-level behavior, and SHA-256 digests;
no local directory, clinical row, identifier, or protected output is included.
The evidence files are not part of this public repository.

## Derived-feature selection

The recoverable completed MIMIC scripts do not establish one rule that agrees
with the maintained specification:

| Evidence file (SHA-256) | Recovered behavior |
|---|---|
| `run_mimic_stage2a_measurements_with_shap.py` (`324349e59f8ddff0e27fb53e7f6f61f042c117dc94408b19f45395ad942e7855`) | Top 50 laboratory concepts; candidates filtered for training support/nonvariance; training median imputation; `mutual_info_classif`; descending mutual information then feature-name tie-break; **15** retained measurement features. |
| `run_mimic_stage2b_medications.py` (`425016627defaf834e905abe3a28d9d8ce6f23334195f4c5034d0f3c6e9f8862`) | Top **250** medication codes; training support/nonvariance filtering; training median imputation; `mutual_info_classif`; descending mutual information then feature-name tie-break; 21 retained features. |
| `run_mimic_stage2c_procedures.py` (`1cb9c6e4a8aa915b9a9da51c5ecca60728741064e5a0a95fc2447dd5701d4d53`) | Top 50 procedure concepts; training support/nonvariance filtering; training median imputation; `mutual_info_classif`; descending mutual information then feature-name tie-break; 21 retained features. |
| `pipeline_specification.pdf` (`984519fdc8e26b010cbea286c79d29c04a769548dd65dbdea3467c7cd34d2e20`) | Top 50 concepts and all 300/104/103 derived candidates; it explicitly rejects interpreting “21” as the number of OOF predictions, but does not specify the later 21-column reduction now required. |

The maintained manuscript material reviewed locally describes training-fold
frequency of occurrence and exactly 21 features per domain. That conflicts with
the completed scripts above. Therefore:

- `training_support_prevalence_v1` is the implemented, deterministic,
  unsupervised rule used by the synthetic demonstration and requested final
  design.
- Support is calculated before imputation from underlying event availability,
  not from stored zero/default values.
- Ties use the deterministic candidate-construction order.
- The paper configurations remain fail-closed until the manuscript and
  completed-script discrepancy is resolved and approved.

This is not a claim that the prevalence rule was used in the completed clinical
analysis.

## MIMIC death ascertainment

`build_mimic_master_cohort.py`
(`3fa5a0bfb133c3545ff2d43c4e0a23b514469c3b3646c3c9b226f164113f9d33`)
normalizes both `admissions.deathtime` and `patients.dod` to dates, takes the
earliest date, excludes death dates on or before the landmark date, and labels
deaths after that date through the 30-day date.

That historical rule discards available time precision and creates the exact
midnight-precedence defect identified in the audit. The adapter now makes the
audited, explicit deviation `precise_admission_deathtime_then_patient_dod_v1`:

- a present `admissions.deathtime` is retained exactly and has priority;
- `patients.dod` is a date-only fallback when no precise death time exists;
- conflicting sources are flagged while the precise value remains selected;
- no clock time is invented for `dod`;
- a date-only death on the landmark calendar date is conservatively excluded;
- the same selected representation drives exclusion, outcome, attrition, and
  hashes.

Clinical-paper mortality reconciliation remains blocked until the corrected
labels are compared with the completed replication.

## MIMIC procedure timing and concepts

`run_mimic_stage2c_procedures.py`
(`1cb9c6e4a8aa915b9a9da51c5ecca60728741064e5a0a95fc2447dd5701d4d53`)
establishes the recovered procedure rule:

- source: `procedures_icd`;
- time field: date-only `chartdate`;
- concept: normalized code namespaced by ICD version (`icd{version}:{code}`);
- inclusion: calendar dates from the admission date through the prediction
  landmark date, inclusive.

The adapter implements this as
`calendar_dates_spanned_inclusive_v1`. It does not convert `chartdate` to
midnight, calculate fictitious hours from admission, or describe the rule as an
exact timestamp-based 24-hour filter.

## SHAP

The recovered stage scripts use model-agnostic permutation SHAP with a
training-fold background and held-out fold evaluation. The most consistent
medication/procedure settings are a 50-row training background, a 100-row
held-out sample, fold-derived fixed seeds, and
`max(2 * transformed_features + 1, 101)` evaluations. The implementation
aggregates encoded columns back to model-input feature names and emits only
fold-level mean absolute SHAP and safe cross-fold summaries. Encounter-level
SHAP values are never written.

No completed unified eight-matrix SHAP script was recovered. Consequently the
code can generate the requested held-out aggregates, but paper SHAP
reconciliation remains blocked.
