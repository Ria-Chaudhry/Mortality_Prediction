# Synthetic freeze history

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
