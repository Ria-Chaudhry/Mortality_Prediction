# Recovered MIMIC-IV method provenance

This document records the completed independent-replication evidence reviewed in
July 2026. It gives only repository-neutral basenames, SHA-256 digests, functions,
and line ranges. The historical files and protected outputs are not copied here,
and no private directory or clinical row is disclosed.

## Evidence register

| Historical evidence (SHA-256) | Section | Recovered rule | Current implementation | Regression evidence |
|---|---|---|---|---|
| `Cohort.py` (`242ad9106b60dc3f228ce6f47d8b460b580fc0da313bd9848541f8adc4141116`) | configuration, lines 22–24 and 61–78 | MIMIC-IV v3.1; seed 42; target 23,000; 365-day lookback; the seven acute admission types listed below; `ELECTIVE` and `SURGICAL SAME DAY ADMISSION` excluded | `configs/mimiciv.paper.yaml`; `adapters/mimic_iv.py` | `test_recovered_mimic_paper_config_passes_source_free_preflight`, native mapping tests |
| `Cohort.py` (same digest) | `normalize_race_ethnicity`, lines 162–202 | combined-race mapping below | `MIMICIVAdapter._native_core` | historical race/ethnicity mapping test |
| `Cohort.py` (same digest) | cohort outcome, lines 297–343 | anchor-year age; calendar-date normalization of `deathtime` and `dod`; earliest nonmissing date; landmark-day exclusion; 30-day label | `_historical_native_deaths`; cohort builder | historical mortality boundary parameterization |
| `Cohort.py` (same digest) | `patient_level_subsample_exact`, lines 388–463 | random patient order with complete groups and at most one partially retained boundary patient | `_historical_patient_subsample` | deterministic subsampling tests and count diagnostics |
| `Baseline.py` (`fd280b0fccabc8047d3daef619f1cee50d0d9aec60af9394365e927d2731d30a`) | `create_patient_grouped_splits`, lines 587–625; saved-fold stage, lines 1443–1471 | patient/start/visit-ordered frozen cohort; `StratifiedGroupKFold`, five folds, shuffle true, seed 42 | `patient_start_visit_v1`; `historical_stratified_group_k_fold_v1` | direct sklearn equivalence, patient isolation, and determinism tests |
| `Measurement.py` (`0ec6f1a8a7a68e4f1a0ddd2413509d53e83ef1e42fec90ca1acdb16fcc32a621`) | extraction, lines 535–651; constants, lines 147–156; selector, lines 947–1029 | direct nonmissing `hadm_id`, `itemid`, and numeric `valuenum` only; `storetime` with `charttime` fallback; 50 laboratory concepts; 21 final columns; training support/nonvariance screen; training median; `mutual_info_classif`; score descending, feature name ascending | `MIMICIVAdapter` direct-HADM/numeric-only paper policies; `features/construction.py` rule `mutual_information_after_training_median_v1` | native extraction, MI fold-isolation, and exact-column tests |
| `Medication.py` (`fc31acc4b6160bca7a4b00c701736fbb7db8086eef2997a470f86f1cdd1864a7`) | extraction, lines 535–643; constants, lines 125–135; `make_medication_code`, lines 488–519; selector, lines 871–949 | direct nonmissing `hadm_id` only; historical code ranked **250** medication codes and retained 21 columns; identifier priority is GSN, then NDC, formulary code, normalized drug name; final selector is training-fold MI with minimum nonzero support | direct-HADM paper policy and identifier hierarchy in `MIMICIVAdapter`; exact requested 50/21 invariant in config/selection | native linkage, hierarchy, support-screen, config rejection, fold-isolation, and exact-count tests |
| `procedure.py` (`0abd028c42a72464625e49a3fb12033b2ce607c114bd8b180489309756e91728`) | constants, lines 135–145; extraction, lines 519–716 | direct nonmissing `hadm_id` only; 50 ICD concepts; 21 columns; `procedures_icd.chartdate`; inclusive admission-date through landmark-date approximation; `icd{version}:{code}` | direct-HADM paper policy, date-only procedure mapping and linkage | native linkage, date fixtures, and boundary tests |
| `run_mimic_pairwise_domain_models.py` (`18dd0ff9cf8a5cb46298cc03f863b870a7634f587e32c2865bf978e83bca1727`) and `run_mimic_full_domain_model.py` (`e79d4d20f75c780b9118877f16ef41b39e9ca1b375caf8fcfc2c16197bfbfd62`) | fold model and explanation stages | reuse saved fold-specific domain columns; permutation SHAP with outer-training background and outer-held-out evaluation | unified pipeline SHAP stage | SHAP partition, order, deterministic, eight-matrix integration tests |
| `mimic_stage2a_analysis_manifest.json` (`fd20e5a8de8fc49a3bc7bda31f255cc43592196c03dbb303079f50cc727c1172`) | measurement extraction and selection manifest | 638,756 numeric availability-window rows; 3,208,546 cohort-HADM rows; 300 candidates; 21 selected; SHAP 100/250 | event-count target, 300/21 assertions, unified SHAP policy | event-count failure diagnostics, exact candidate/selection tests |
| `mimic_stage2b_analysis_manifest.json` (`ef034349eb93694649d6285b3a99de7e0f0ab4bea6501171176cf5a0cd0f6af5`) | medication extraction and selection manifest | 374,658 window rows; 828,623 cohort-HADM rows; historical 504 candidates from 250 codes; 21 selected; SHAP 50/100 | event-count target, corrected requested 104 candidates from 50 codes, unified SHAP policy | event-count failure diagnostics, 50/21 rejection tests |
| `mimic_stage2c_analysis_manifest.json` (`0ec6fc560361d8a4c5574bc4077f17308bd4df66ff04c1a049c06351e4c980ee`) | procedure extraction and selection manifest | 17,130 calendar-window rows; 32,277 cohort-HADM rows; 104 saved candidates with inverse missingness excluded from selection; SHAP 50/100 | event-count target, requested 103 nonredundant candidates, unified SHAP policy | procedure boundary/count and 50/21 tests |

## Final feature-selection rule

The recovered final-column selector is
`mutual_information_after_training_median_v1`:

1. Use the outer-training visits only.
2. Apply the historical training support and nonvariance eligibility screens.
3. Fit median imputation on eligible outer-training values.
4. Calculate `mutual_info_classif` against outer-training outcomes with the
   fold-derived fixed seed.
5. Rank by mutual information descending, then feature name ascending.
6. Retain exactly 21 derived columns.

The support screen is domain-specific exactly as recovered: medication
features require at least 50 nonzero/observed training visits (not 50 visits in
both binary classes); measurement missingness and procedure presence flags
require both values to have the configured minimum support; count and
availability features use positive/nonmissing training support.

Twenty-one means model-matrix columns, not concepts. Multiple summaries from one
concept may be selected. The restricted candidate table records the source
concept, summary type, pre-imputation support, MI score, tie-break value, rank,
eligibility reason, rule/version, and selected flag.

The completed medication script used 250 candidate codes. The final requested
analysis explicitly requires exactly 50 concepts in every domain, so the current
pipeline deliberately overrides that historical medication constant. It also
requires concept selection inside each outer training fold. Configuration,
runtime validation, artifact verification, and tests reject 250 concepts or 15
final features. This override is transparent and is one reason that a new
clinical run must be reconciled before any paper-reproduction claim.

The synthetic demonstration continues to use the configured unsupervised
`training_support_prevalence_v1` rule so leakage and tied-support behavior are
independently testable. Paper MIMIC mode freezes the recovered MI rule.

## Dataset, admission, race, and ethnicity rules

The completed pipeline names MIMIC-IV **v3.1**.

Acute/non-elective admission types are:

- `AMBULATORY OBSERVATION`
- `DIRECT EMER.`
- `DIRECT OBSERVATION`
- `EU OBSERVATION`
- `EW EMER.`
- `OBSERVATION ADMIT`
- `URGENT`

`ELECTIVE` and `SURGICAL SAME DAY ADMISSION` are excluded.

The combined `admissions.race` field is uppercased and stripped. Hispanic/Latino
strings map ethnicity to `HISPANIC_OR_LATINO` and race to `OTHER`. Unknown,
unable, declined, not-specified, other/unknown, and Portuguese strings map both
audit categories to unknown. Remaining ethnicity is
`NOT_HISPANIC_OR_LATINO`. Race recognizes White, Black/African, Asian,
American Indian/Alaska Native, Native Hawaiian/Pacific Islander, and Multiple;
other values map to `OTHER`.

The historical minimum age was 18. The current task explicitly changes the
MIMIC minimum to **0**. That is a documented task-level deviation, not a
historical claim, and its effect must be included in real-run reconciliation.

## Medication identifier

The historical medication identifier is a deterministic fallback hierarchy:

```text
normalized GSN
→ normalized NDC
→ normalized formulary_drug_cd
→ normalized drug name
```

Concepts are namespaced as `prescriptions:gsn:…`,
`prescriptions:ndc:…`, `prescriptions:formulary:…`, or
`prescriptions:drug:…`. Empty strings, `nan`, `none`, `0`, and `0.0` are treated
as unavailable. This is not a fabricated drug-normalization ontology.

The completed extraction dropped medication, measurement, and procedure rows
without a qualifying `hadm_id`; it did not infer a paper event link from patient
and time. Paper mode therefore freezes `direct_hadm_only_v1`. The generic
native adapter retains a separately configured patient-time fallback for
nonpaper layouts that require and test it.

The completed measurement and prescription scripts compared event availability
directly with admission time and the 24-hour prediction time; they did not
shorten that interval at an earlier discharge time. MIMIC paper mode therefore
uses `admission_plus_window_v1`. The alternative discharge-capped policy
remains explicit for configurations that require it.

## Mortality rule

The historical cohort normalized `admissions.deathtime` and `patients.dod` to
calendar dates, chose the earliest nonmissing date, excluded a death date on or
before the normalized 24-hour landmark date, and labeled death after that date
through the normalized admission-plus-30-day date.

Paper MIMIC mode reproduces that date-granularity rule exactly and records
source disagreement. It does not claim timestamp precision and does not allow a
midnight-coerced date to masquerade as a precise time. Thus a precise death at
26 hours and `dod` on the same calendar date are represented by that calendar
date and excluded under the historical method. The separate generic native
adapter rule remains available for nonpaper use and preserves precise
`deathtime`; tests protect both explicit policies.

## Procedure timing and concepts

The completed analysis used `procedures_icd`, date-only `chartdate`, and
`icd{version}:{punctuation-normalized code}`. It included calendar dates from
the admission date through the normalized 24-hour landmark date, inclusive.
The implementation describes this honestly as a calendar-date approximation.
It does not invent midnight timestamps or exact admission-relative hours.

## Subsampling and reconciliation targets

`np.random.default_rng(42)` permutes unique patients. Complete patient groups
are accumulated until 23,000 admissions; if the next group exceeds the target,
only the remaining randomly shuffled visits from that one boundary patient are
retained. Output is sorted by patient, admission time, and admission ID. The
completed baseline then applies five-fold `StratifiedGroupKFold` with shuffle
and seed 42 to that frozen order; paper mode reproduces that method rather than
the generic grouped-fold policy.

The historical saved aggregate and the confirmed reconciliation target are
23,000 admissions, **10,006** patients, and 819 deaths. The repository does not
force those values: paper execution persists full expected-versus-observed
attrition diagnostics and fails at the first mismatch.

## Unified SHAP policy

One authoritative pipeline stage now processes the selected model for each of
all eight matrices and all five outer folds. Model choice uses the frozen
AUPRC, AUROC, Brier, then model-order hierarchy. For each fold/matrix, the stage
deterministically reconstructs the selected fold fit and first verifies its
probabilities against the stored OOF rows. It then uses permutation SHAP with:

- a deterministic outer-training background (maximum 50 rows);
- a deterministic outer-held-out evaluation sample (maximum 100 rows);
- exact fitted preprocessing and feature order;
- seed `42 + outer_fold`;
- `max(2 × transformed feature count + 1, 101)` evaluations.

The completed stage scripts used heterogeneous sample caps (baseline 100/500,
measurements 100/250, and medications/procedures/pairwise 50/100) and did not
offer one eight-matrix entry point. The unified analysis freezes the common
50-training/100-held-out policy for every matrix so sampling and aggregation do
not vary by matrix. This is a documented consolidation, not a claim that the
fragmented historical SHAP tables were byte-for-byte generated by this stage.

Encoded contributions are summed back to the input feature column. Fold output
is mean absolute SHAP only. Cross-fold output averages a feature over folds in
which that fold-specific feature was selected; absence is not silently treated
as zero. Encounter-level SHAP values are never written.

The historical scripts did not provide one single eight-matrix entry point, so
the unification is an implementation consolidation of their shared held-out
procedure. Manuscript reconciliation remains false until a protected-data run
is compared with the completed outputs.
