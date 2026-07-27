# Expected aggregate demonstration

`make synthetic-run` creates aggregate results for each adapter under `outputs/synthetic/`.
`make verify` compares cohort and design counts to
`synthetic_data/expected_outputs/expected_summary.json`, checks every recorded SHA-256 output
checksum, and validates the required analytical schemas. Visit-level predictions are written only
to the ignored `restricted_outputs/synthetic/` directory.
