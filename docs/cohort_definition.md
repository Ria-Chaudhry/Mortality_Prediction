# Cohort and outcome definition

The unit is one configured parent acute-care encounter. Eligible visits are adult, acute,
non-elective hospital/observation encounters. Configuration defines the accepted visit values and
age range. Row order is deterministic by start time, patient key, and visit key; the resulting
`cohort_visit_number` is frozen.

The prediction landmark is `visit_start + 24 hours`. The predictor interval is
`[visit_start, min(visit_end when known, landmark))`. Its right boundary is exclusive. Short
visits remain when configured, with the predictor interval ending at visit end. Death on or before
the landmark excludes the visit.

The outcome is one only when death is after the landmark and on or before 30 days from admission.
A non-event requires verified survival through the horizon using a later death or a follow-up end
on/after the horizon.

Baseline predictors are age, sex, race, ethnicity, acute visit type, prior visit count, prior
acute visit count, a prior-visit indicator, and a non-age-adjusted Charlson score. Charlson
conditions are taken only from qualifying prior acute encounters beginning in the 365-day
lookback; current-admission billing diagnoses are never joined.

Identifiers, timestamps, deaths/outcomes, discharge information, length of stay, predictions,
post-landmark values, and future-derived fields are hard-failed if presented as predictors.
