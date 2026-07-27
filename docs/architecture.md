# Architecture

The dependency direction is one-way:

```text
source files or read-only SQL
        |
CHoRUSAdapter / MIMICIVAdapter
        |
standardized tables + mapping/input hashes
        |
frozen cohort -> patient folds -> linked first-24-hour events
        |
per-fold concept selection -> per-fold domain features -> eight matrices
        |
training-only preprocessing -> four models -> held-out probabilities
        |
pooled OOF evaluation -> aggregate tables/coordinates -> manifests
```

`adapters/` is the only package that knows physical source table names. `cohort/`, `features/`,
`modeling/`, `evaluation/`, and `audit/` accept only standardized frames or outputs produced from
them. A run is one dataset; no function accepts a second dataset or a previously trained model.

The public CLI invokes `pipeline.run_pipeline`. Numbered scripts are thin stage wrappers around
the same function. Re-running a stage rebuilds its prerequisites, preventing stale discovery from
silently becoming analysis.

Public and restricted output roots are separate at creation time. Identifiers, dates, event rows,
fold rows, OOF predictions, and protected feature names never need to enter public results.
