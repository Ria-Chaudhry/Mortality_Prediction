# Security and privacy

Artifact classes are:

- `restricted`: source-derived rows, identifiers, dates, mappings requiring protection,
  folds, features, OOF predictions, fitted objects, and unit audits by default;
- `release_candidate_aggregate`: clinical aggregates awaiting governance review;
- `public_clinical`: explicitly approved clinical aggregates that passed the actual allowlist and
  configured small-cell gate;
- `public_synthetic`: privacy-safe synthetic aggregates and audits.

Real clinical output defaults to restricted. Publication requires an explicit release approval,
approved small-cell threshold, allowlisted file/schema, and a manifest record that the
`public_clinical` gate actually ran and passed. The scanner rejects identity/date
columns, row predictions or feature values, private paths, usernames, mount/server/connection
details, credentials, tokens, keys, unexpected schemas, and cells below the approved threshold.
It logs only finding types and columns, not clinical values.

`measurement_unit_audit.csv` is public for synthetic execution. Clinical unit audits are
restricted unless small-cell handling and explicit release approval permit an allowlisted copy.

Raw CHoRUS/MIMIC data, clinical identifiers, assignments, OOF rows, event samples, `.env`, keys,
connections, logs, and model objects belong outside Git under ignored restricted storage.
Environment-variable names—not credentials—appear in configuration.
