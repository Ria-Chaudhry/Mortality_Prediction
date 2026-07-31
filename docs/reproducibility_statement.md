# Reproducibility statement

The repository executes and verifies a privacy-safe synthetic analysis through both adapters.
That establishes code-path reproducibility for the synthetic inputs, not reproduction of the
paper's protected results.

Canonical hashes cover analytical values, source snapshot identifiers, configurations, mappings,
cohort order, folds, concept/derived selections, feature schemas/values, matrices, and outputs.
Fixed ordering and seeds govern folds, ties, estimators, top-risk groups, and bootstrap samples.
Timestamps, machine paths, and Git dirty state are excluded from canonical synthetic equality.
Raw serialization hashes remain enforced within each run. Public numeric tables are serialized
and canonically compared at ten decimal places. Derived floating-point features are canonically
rounded to eight decimal places
before feature hashing, preprocessing, and fitting; this versioned configuration boundary removes
platform-level aggregation noise before it can change tree split ties.

Pinned scikit-learn random-forest and gradient-boosting builds can nevertheless choose different
tied splits across operating systems. Exact frozen fitted-model acceptance is therefore restricted
to CPython 3.10.13 on Linux x86_64, with CI pinned to Ubuntu 24.04. The committed freeze records
the Python version, platform, architecture, and dependency-lock hash. Other platforms may execute
the synthetic pipeline, verify each dataset's exact same-run hashes and structural invariants, and
run repeated-process equality tests, but parent-level frozen verification fails closed rather than
certifying a different fitted-model baseline.

The synthetic configuration lowers bootstrap repetitions for runtime while exercising the same
patient-clustered percentile implementation. Paper configuration specifies 2,000.

The completed MIMIC source identifies release v3.1 and establishes its
admission, race/ethnicity, medication, mortality, measurement availability,
procedure-date, direct-HADM extraction, predictor-end, subsampling, row-order,
and stratified patient-fold rules. Confirmed CHoRUS mapping/snapshot, real run
reconciliation, and release-cleared clinical aggregates remain unavailable. No
real result is fabricated.

Recovered completed MIMIC scripts establish 21 final derived columns selected by outer-training
mutual information after training-median imputation. The completed medication stage used 250
concepts; the corrected final design requires 50 for every domain. Paper mode encodes that
explicit override and the historical date-normalized mortality rule. These choices require
protected-output reconciliation; see `recovered_method_provenance.md`.

Internally selected models are described from the same OOF predictions used for selection; they
are not independent test estimates. This code does not establish clinical safety, causal effect,
transportability, or deployment readiness.
