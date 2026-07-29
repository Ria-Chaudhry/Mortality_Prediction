# Reproducibility statement

The repository executes and verifies a privacy-safe synthetic analysis through both adapters.
That establishes code-path reproducibility for the synthetic inputs, not reproduction of the
paper's protected results.

Canonical hashes cover analytical values, source snapshot identifiers, configurations, mappings,
cohort order, folds, concept/derived selections, feature schemas/values, matrices, and outputs.
Fixed ordering and seeds govern folds, ties, estimators, top-risk groups, and bootstrap samples.
Timestamps, machine paths, and Git dirty state are excluded from canonical synthetic equality.
Raw serialization hashes remain enforced within each run but are excluded from the committed
cross-platform manifest; public numeric tables are serialized and canonically compared at ten
decimal places.
Frozen aggregate verification supports CPython 3.10.13 exactly; unsupported runtimes fail before
execution rather than producing a misleading cross-runtime manifest mismatch.

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
