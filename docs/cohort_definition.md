# Cohort and outcome definition

The unit is one configured parent acute-care encounter. Configured age, acute, and non-elective
eligibility is applied before feature selection. Row order is stable by start time, patient key,
and visit key. The MIMIC paper configuration uses minimum age 0 by explicit task requirement;
the completed historical script used 18, so this deviation is reconciled in real runs.

The prediction landmark is `start + landmark_hours` (24 hours in this design).
The predictor interval is independently controlled by
`predictor_window_hours` and a versioned end policy:

```text
admission_plus_window_v1:          [start, start + predictor_window_hours)
earliest_discharge_or_window_v1:   [start, min(end when known, start + predictor_window_hours))
```

The right edge is exclusive. MIMIC paper mode uses
`admission_plus_window_v1`, matching the completed scripts even when discharge
precedes hour 24. Other configurations can explicitly select the discharge cap.
Configuration rejects a predictor window after the landmark unless an explicit
documented override permits it.

Death on or before the landmark excludes the visit. Outcome is one for death after the landmark
and on or before 30 days from start. MIMIC paper mode reproduces the completed date-level rule:
normalize both `admissions.deathtime` and `patients.dod`, select the earliest nonmissing date,
exclude on/before the normalized landmark date, and label through the normalized 30-day date.
Source disagreement is audited. A separate generic native rule preserves precise `deathtime` and
uses `dod` only as a date fallback; the two policies cannot be mixed silently. A zero requires
the configured explicit follow-up policy.

Charlson and utilization use qualifying prior encounters beginning on or after
`index_start - 365 days` and strictly before index start. Index-admission diagnoses are excluded.
See `docs/charlson.md`.

Identifiers, timestamps, outcome/death fields, discharge information, length of stay,
predictions, and post-landmark values are prohibited model columns.
