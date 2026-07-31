# Architecture

```text
projected source tables / read-only SQL
        |
adapter cohort-first relation/scan -> candidate acute encounters
        |
cohort/patient/time-restricted diagnoses and domain scans
        |
standardized tables + canonical value signatures
        |
frozen cohort -> frozen patient folds -> linked predictor-window events
        |
outer-training top 50 concepts -> candidate columns
        |
outer-training configured derived selector -> exactly 21 columns/domain -> eight matrices
        |
training-only preprocessing/model -> held-out probabilities
        |
verified OOF-equivalent selected fold fit -> unified held-out permutation SHAP
        |
OOF fold-identity validation -> aggregate analyses -> classified artifacts
```

Only `adapters/` knows physical tables. CHoRUS uses a server-side eligible-cohort relation,
configured projections, and SQL cohort/patient/window predicates without cohort-sized parameter
expansion.
MIMIC uses Parquet pushdown or projected chunked CSV reads. Common cohort, feature, model,
evaluation, and audit code consumes standardized frames.

One run accepts one dataset. The same fold/domain feature object is reused by every matrix
containing that domain. There is no transfer of outcomes, concepts, folds, preprocessing, or
models between CHoRUS and MIMIC.

MIMIC paper mode freezes the completed patient/start/visit row order and
five-fold shuffled `StratifiedGroupKFold` with seed 42. The generic grouped-fold
method is a separate configuration value rather than an implicit substitute.

Real output roots are restricted by default. A clinical aggregate becomes `public_clinical` only
after an actual allowlist/small-cell scan runs and passes with explicit approval. Synthetic
aggregate output is `public_synthetic`.
