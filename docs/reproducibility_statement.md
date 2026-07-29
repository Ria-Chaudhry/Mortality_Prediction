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

The exact MIMIC release, confirmed CHoRUS mapping/snapshot, approved unit conversions, real
manifests, manuscript reconciliation, and release-cleared aggregates are unavailable. Paper
configurations fail closed on those fields. No real result is fabricated.

Recovered completed MIMIC scripts conflict with maintained manuscript material on derived-feature
selection, measurement feature count, and medication concept count. The historical date-normalized
death rule also conflicts with the precision correction required by the audit. Those
methodological reconciliations remain explicit paper-run blockers; see
`recovered_method_provenance.md`.

Internally selected models are described from the same OOF predictions used for selection; they
are not independent test estimates. This code does not establish clinical safety, causal effect,
transportability, or deployment readiness.
