# Methods-to-code and output crosswalk

| Method or figure/table input | Configuration | Code/stage | Evidence/output |
|---|---|---|---|
| Adult non-elective acute-care eligibility | `cohort.yaml` | `cohort/builder.py`, stage 2 | attrition, cohort manifest |
| Predictor window vs 24-hour landmark | separate hour fields | builder + `features/linkage.py` | cohort/linkage audits |
| Early-death exclusion and 30-day outcome | horizon/follow-up policy | builder | restricted cohort, attrition |
| Prior utilization | 365-day lookback | `_prior_features` | restricted baseline |
| Versioned prior-only Charlson | algorithm/version/hash | `cohort/charlson.py` | baseline, config/mapping manifest |
| Five patient folds | count/seed | `cohort/folds.py`, stage 3 | fold summary/hash/assignments |
| Authoritative direct/missing-only bridge and patient-time linkage | adapter mapping | `features/linkage.py` | public aggregate plus restricted reason audit |
| Outer-training top 50 | count/rank/tie | `features/selection.py` | concept selections/hashes |
| 300/104/103 candidate features | domain definitions | `features/construction.py` | constructed count |
| Final outer-training 21 columns | support proportion/construction-order tie | construction | all-candidate selection table; 21 selected/domain/fold |
| Equal domain counts/reuse across matrices | matrix definitions | `features/matrices.py` | matrix/feature manifests and tests |
| Four frozen models/eight matrices | `models.yaml` | `modeling/runner.py` | 160 fit manifests |
| Training-only learned transforms | model pipeline | `fit_predict_fold` | tests/fit manifests |
| Frozen fold identity for every OOF row | fold assignments | OOF validator | restricted OOF and adversarial tests |
| Fold/pooled metrics | threshold config | `evaluation/analysis.py` | metric CSVs |
| Patient bootstrap CIs | bootstrap config | `evaluation/bootstrap.py` | interval CSV |
| Best model per matrix | frozen hierarchy | evaluation | `best_model_by_matrix.csv` |
| ROC and 90% specificity | specificity target | metrics | ROC/operating-point CSVs |
| Highest-risk 10% | risk fraction/name tie | metrics | top-risk CSV |
| Calibration | bin count | metrics/analysis | summary/coordinate CSVs |
| Decision curves | threshold sequence | analysis | decision coordinates |
| Paired domain increments | comparison list/shared samples | analysis | paired comparisons |
| Performance table | selected models | stage 8 | performance table |
| Clinical-utility table | operating point/top-risk | stage 8 | utility table |
| Calibration table | selected calibration | stage 8 | calibration table |
| Held-out SHAP | permutation/training background/held-out sample | `modeling/shap_analysis.py` | restricted fold aggregate and safe cross-fold summary |
| Plot inputs | saved numerical coordinates | stage 8 | ROC/calibration/decision CSVs; no plots |
| Paper cohort/attrition/event/selection counts | paper YAML only | builder/pipeline gates | stage comparisons and failed manifest on mismatch |
| Paper reconciliation/release | paper YAML/governance | nonconnecting preflight and `verify_paper_run` | recomputed restricted evidence plus actual `public_clinical` gate |

The completed MIMIC scripts and maintained manuscript disagree on the selection rule, concept
count for medications, and measurement feature count. Paper configurations fail closed pending
reconciliation; `recovered_method_provenance.md` records the evidence.
