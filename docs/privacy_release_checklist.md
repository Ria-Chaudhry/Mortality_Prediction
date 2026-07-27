# Privacy release checklist

Before staging a clinical run:

1. Confirm only source, configuration without secrets, documentation, synthetic data, and
   disclosure-cleared aggregates are candidates.
2. Confirm `restricted_outputs/`, raw roots, `.env`, models, logs, and credentials are ignored.
3. Inspect `git status --short` and the complete staged diff.
4. Search staged text for identifiers, dates, patient rows, connection URLs, tokens, hostnames,
   private paths, and license declarations.
5. Recompute output checksums and run the public-tree privacy scanner.
6. Apply the source data-use agreement's small-cell review to every aggregate.

The automated scanner is deliberately conservative and is not a replacement for human disclosure
review.
