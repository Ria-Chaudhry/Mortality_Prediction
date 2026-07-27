# Feature data dictionary

## Baseline

| Field | Definition |
|---|---|
| `age` | Years at admission; no age component is added to Charlson. |
| `sex`, `race`, `ethnicity` | Source-normalized categorical values. |
| `visit_type` | Configured acute encounter category. |
| `prior_visit_count` | Distinct prior encounters in the 365-day lookback. |
| `prior_acute_visit_count` | Distinct prior configured acute encounters in lookback. |
| `prior_visit_indicator` | One when any prior encounter exists. |
| `prior_charlson_score` | Sum of category weights from prior acute encounters only. |

## Measurements (300 columns per fold)

The 50 training-selected usable numeric concepts each contribute `mean`, `max`, `min`, sample
`sd` (`ddof=1`, zero for one value), valid-value `count`, and `missing`. With no valid value,
summaries stay missing until training median imputation, count is zero, and missing is one.
Categorical results and unconfirmed/incompatible units are not coerced. Confirmed conversions are
configuration-driven; a training-mode unit option, when explicitly chosen, learns the unit only
from training visits.

## Medications (104 columns per fold)

Each of 50 training-selected concepts contributes qualifying-record exposure and count.
`any_drug_24h`, `unique_drug_count_24h`, `repeat_drug_exposure_count_24h`, and
`time_to_first_drug_in_hours` use every qualifying concept, not only selected concepts. Repeat
count is the sum of records beyond the first within each visit-concept. Time is missing with no
drug.

## Procedures (103 columns per fold)

Each of 50 training-selected concepts contributes qualifying-record exposure and count.
`any_procedure_24h`, `unique_procedure_count_24h`, and `procedure_count_total_24h` use every
qualifying concept.

Every feature frame retains every frozen visit and the same row order. Medication/procedure zeros
mean no qualifying record, not a claim that no care occurred. Source semantics are retained in
selection and mapping audits.
