# Reproducibility statement

The repository reproduces the specified method with public synthetic data. It does not redistribute
or reconstruct protected CHoRUS or MIMIC-IV records. Exact clinical results require the authorized
source release, confirmed mappings, frozen configuration, code commit, and locked environment
recorded in the run manifest.

Material inputs, configuration, mappings, cohort, row order, folds, selections, features, and
public outputs are SHA-256 hashed. Fixed seeds and stable ordering govern folds, concept ties,
top-risk ties, bootstrap samples, preprocessing, and estimators. A repeated analytical run with
the same inputs/configuration/software must reproduce analytical outputs; wall-clock timestamps
are provenance and are excluded from analytical equality.

The synthetic run uses fewer bootstrap repetitions for practical CI time but exercises the same
patient-clustered percentile implementation. Paper configuration retains the prespecified 2,000
repetitions.

Internally selected models use the same pooled OOF predictions for selection and description.
They are not independent test estimates. Operating thresholds are descriptive. This software does
not establish causal effects, clinical safety, transportability, or deployment readiness.
