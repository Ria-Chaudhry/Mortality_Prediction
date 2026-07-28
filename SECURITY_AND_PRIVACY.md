# Security and privacy

Artifact classes are:

- `internal_restricted`: source-derived rows, identifiers, dates, mappings requiring protection,
  folds, features, OOF predictions, fitted objects, and unit audits by default;
- `release_candidate_aggregate_restricted`: clinical aggregates awaiting governance review;
- `public_synthetic`: privacy-safe synthetic aggregates and audits.

Real clinical output defaults to restricted. Publication requires an explicit release approval,
approved small-cell threshold, and allowlisted file/schema. The scanner rejects identity/date
columns, row predictions or feature values, private paths, usernames, mount/server/connection
details, credentials, tokens, keys, unexpected schemas, and cells below the approved threshold.
It logs only finding types and columns, not clinical values.

`measurement_unit_audit.csv` is public for synthetic execution. Clinical unit audits are
restricted unless small-cell handling and explicit release approval permit an allowlisted copy.

Raw CHoRUS/MIMIC data, clinical identifiers, assignments, OOF rows, event samples, `.env`, keys,
connections, logs, and model objects belong outside Git under ignored restricted storage.
Environment-variable names—not credentials—appear in configuration.
