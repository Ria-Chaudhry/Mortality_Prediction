# Contributing

Open an issue before a substantial change. Pull requests must preserve the source-neutral adapter
contract, patient-grouped folds, training-fold-only concept selection, protected/public output
boundary, and deterministic synthetic run. Run `make lint`, `make test`, `make synthetic-run`, and
`make verify` before requesting review.

Never add clinical rows, identifiers, credentials, connection strings, internal paths, or
visit-level predictions. Any new source semantics must be explicit and tested. Scientific changes
must update configuration, the Methods-to-code crosswalk, and the pipeline specification history.
