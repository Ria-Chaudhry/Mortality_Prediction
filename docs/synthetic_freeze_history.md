# Synthetic freeze history

## July 29, 2026 derived-float portability boundary

Clean Ubuntu and macOS executions with the same CPython and dependency lock
showed that platform-level floating aggregation noise could change tied
scikit-learn tree splits. LightGBM outputs were identical; the affected public
feature/matrix hashes and random-forest/gradient-boosting aggregates established
that the divergence entered through derived numeric values before fitting.

The pipeline now applies the versioned
`derived_numeric_decimal_round_v1` rule at eight decimal places to finite
derived floating features before support ranking, hashing, preprocessing, or
model fitting. Missingness and integer occurrence features are unchanged. Two
complete final-code runs produced identical child run IDs and raw output-hash
maps with digest
`aba3d8674f8717e3ec5b1b247e47d4d7ecea2b5981efe5c85eb71da62d773c2a`
before the guarded freeze command was used.

## July 29, 2026 post-implementation provenance refresh

The final clean-checkout verification correctly rejected safe-manifest hashes
that still contained the package code hash from before the last implementation
edits. All canonical analytical artifact hashes were unchanged. Two complete
synthetic executions of the final code produced identical child run IDs and
identical raw output-hash collections; the digest of those run IDs and output
hash maps was
`972ef695ec13630abb7f6301dd78851b26b36ebb7903e99eff0c44dbfcc86a17`.

The guarded freeze command therefore updated only the two dataset
safe-run-manifest hashes, the parent safe-manifest hash, and the expected-file
checksum. No pooled metric, fold metric, calibration, utility, decision-curve,
paired-comparison, selection, SHAP, or other analytical artifact hash changed.

## July 29, 2026 integrity and portability correction

The format-3 freeze replaces cross-runtime raw-byte pins with canonical hashes
of the actual published tables. Exact raw checksums are still verified inside
each run. Public floating-point cells are serialized at ten decimal places,
and the committed comparison preserves file set, column order, row order,
nonfinite-coordinate semantics, and every value at that published precision.

The reviewed analytical/provenance changes in this freeze are:

- feature-value hashes now preserve row identities and column order;
- standardized prior encounters keep Charlson lookback data after cohort-first
  SQL restriction;
- selection hashes include training-visit counts, and verification recomputes
  every concept/derived evidence field plus fold feature tables and matrices;
- model manifests hash frozen fit partitions and fitted preprocessing state;
- mapping validation publishes schema hashes rather than identifier-bearing
  column lists;
- linkage audits distinguish authoritative explicit visits from missing-only
  bridge/patient-time fallback;
- run verification enforces the complete artifact set and reruns the recorded
  privacy classification.

Two clean runs under CPython 3.10.13 and the committed dependency lock produced
identical raw output hashes, child run IDs, and safe manifests for both
adapters. Their canonical collection digest was
`401a13475443749296bd0f232a86d99e2b116b073972850fb8a8157d0f181793`
across 83 files (41 per dataset plus the overall summary).

## July 2026 analytical correction

The expected aggregate hashes changed for reviewed reasons:

- derived-feature selection artifacts now contain all 300/104/103 candidates,
  explicit pre-imputation training support, tie-break values, and exactly 21
  selected final columns;
- MIMIC procedures use genuine date-only `chartdate` and the recovered
  inclusive calendar-date-span rule;
- death precision/source fields and their safe audit counts enter input/cohort
  signatures;
- selected-model held-out permutation-SHAP fold aggregates and summaries are
  now frozen;
- runtime and privacy-gate fields are explicit in safe manifests.

Before updating the pins, two clean runs under the locked CPython 3.10.13
environment produced identical child run IDs and identical deterministic
artifact collection digest
`b51be84e902e42424a20dd3b8a4fe44c7aabd77045170be64c1254571804f89e`.
Timestamp-bearing run manifests were excluded from that comparison exactly as
they are excluded from canonical verification.

Expected outputs may be updated only after review by running:

```bash
make freeze-synthetic-expected
```
