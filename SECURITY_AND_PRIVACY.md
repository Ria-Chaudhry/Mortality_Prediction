# Security and privacy

Raw data, source identifiers, dates, event-level audits, patient-to-fold assignments, OOF
predictions, fitted artifacts, and logs with patient values belong only in `restricted_outputs/`
or another access-controlled location outside this repository. That directory is ignored.

Public outputs are aggregates and plot-ready coordinates. Before publication, users remain
responsible for their data-use agreement, small-cell policy, and disclosure review. Database
access is read from environment variables; configuration must contain only the variable name.

Report a suspected disclosure privately to the repository owner. Do not open a public issue
containing sensitive values.
