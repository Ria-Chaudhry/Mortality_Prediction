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

## Requested top-21 derived-feature override

The 300/104/103 candidate columns are ranked before imputation using outer-training visits only:

- measurement mean/max/min/SD are available when nonmissing;
- a measurement count occurs when positive;
- a measurement missing flag is counted as available when zero, meaning the concept was observed;
- medication/procedure exposure, count, and numeric aggregates occur when positive;
- time-to-first-drug is available when nonmissing.

Rank is descending training-visit occurrence/availability, then ascending exact feature name.
Exactly 21 columns per domain are retained and applied unchanged to validation visits. The same
fold/domain list is reused in every matrix containing that domain. The audit records rank,
training occurrence, denominator, definition, and selection hash.

This rule is unsupervised and contains no outcome, validation frequency, missingness, value, unit
choice, or preprocessing information. It was not specified by the finalized PDF or historical
analysis code; it is an explicit methodological override required by the current implementation
request.
