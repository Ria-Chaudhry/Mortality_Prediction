# Cohort and outcome definition

The unit is one configured parent acute-care encounter. Adult, acute, non-elective eligibility is
applied before feature selection. Row order is stable by start time, patient key, and visit key.

The prediction landmark is `start + landmark_hours` (24 hours in this design). The predictor
interval is independently controlled by `predictor_window_hours`:

```text
[start, min(end when known, start + predictor_window_hours))
```

The right edge is exclusive. A short retained visit stops collection at its end. Configuration
rejects a predictor window after the landmark unless an explicit documented override permits it.

Death on or before the landmark excludes the visit. Outcome is one for death after the landmark
and on or before 30 days from start. A zero requires the configured verified-follow-up rule.
Native MIMIC follow-up policy must be explicit; paper mode will not infer it.

Charlson and utilization use qualifying prior encounters beginning on or after
`index_start - 365 days` and strictly before index start. Index-admission diagnoses are excluded.
See `docs/charlson.md`.

Identifiers, timestamps, outcome/death fields, discharge information, length of stay,
predictions, and post-landmark values are prohibited model columns.
