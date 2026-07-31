# Feature data dictionary

## Baseline

| Field | Definition |
|---|---|
| `age` | Years at admission; MIMIC uses its anchor-year formula. No age weight enters Charlson. |
| `sex`, `race`, `ethnicity` | Explicitly normalized categorical values. |
| `visit_type` | Configured acute encounter category. |
| `prior_visit_count` | Distinct prior encounters in the lookback. |
| `prior_acute_visit_count` | Distinct prior configured acute encounters in the lookback. |
| `prior_visit_indicator` | One when any prior encounter exists. |
| `prior_charlson_score` | Hierarchy-adjusted non-age score from prior acute admissions only. |

## Top-50 concept construction

Within each outer training fold, concepts are ranked by distinct training acute-care visits with
at least one qualifying record. Ties use normalized concept key. Outcomes are not read.

- Measurements: 50 concepts x mean, maximum, minimum, sample SD, valid count, missing flag = 300.
- Medications: 50 x exposure/count plus four all-concept aggregates = 104.
- Procedures: 50 x exposure/count plus three all-concept aggregates = 103.

Measurement summaries remain missing until training-fold median imputation. SD uses `ddof=1` and
is zero for one value. Incompatible or nonnumeric measurement rows do not become values. Drug and
procedure zeros mean no qualifying record, not proof that care was absent.

## Final 21 derived model columns

The 300/104/103 candidates carry pre-imputation support evidence calculated from outer-training
visits only. Twenty-one means final matrix columns, not concepts; several summaries from one
concept may be selected:

- measurement mean/max/min/SD are available when nonmissing;
- a measurement count occurs when positive;
- a measurement missing flag is counted as available when zero, meaning the concept was observed;
- medication/procedure exposure, count, and numeric aggregates occur when positive;
- time-to-first-drug is available when nonmissing.

The synthetic rule ranks support proportion descending, then candidate-construction order. The
MIMIC paper configuration freezes the recovered completed-analysis rule: apply training
support/nonvariance eligibility, fit median imputation on training values, score
`mutual_info_classif` against training outcomes, and rank by score descending then feature name
ascending. No validation value, outcome, missingness pattern, unit, or preprocessing fit enters
either rule.

Exactly 21 columns per domain are retained and applied unchanged to validation visits. The same
fold/domain list is reused in every matrix containing that domain. The audit records rank,
source concept, summary type, training support count/proportion, selection score, explicit
tie-break value, eligibility reason, rule/version, selected status, and selection hash for every
candidate. The historical medication stage used 250 concepts; the required corrected design
overrides it to 50. See `recovered_method_provenance.md`.
