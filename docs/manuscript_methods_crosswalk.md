# Methods-to-code and output crosswalk

| Manuscript/specification element | Configuration | Generating code/stage | Output |
|---|---|---|---|
| Adult non-elective acute-care population | `cohort.yaml: cohort` | `cohort/builder.py`, stage 2 | `attrition.csv`, cohort manifest |
| 24-hour landmark and exclusive window | `landmark_hours`, `predictor_window_hours` | cohort builder + `features/linkage.py` | event linkage audit |
| Early-death exclusion and 30-day outcome | `outcome_horizon_days` | cohort builder | restricted cohort, cohort manifest |
| Verified follow-up | `require_verified_followup` | cohort builder | attrition |
| Prior utilization | `prior_lookback_days` | `_prior_features` | restricted baseline |
| Prior-only non-age Charlson | `charlson_mapping.csv` | `_prior_features` | restricted baseline |
| Five patient folds | `folds.count`, `folds.seed` | `cohort/folds.py`, stage 3 | fold summary/hash; restricted assignments |
| Direct/bridge/patient-time linkage | dataset mapping | `features/linkage.py`, stages 4-5 | linkage/mapping audits |
| Training-fold concept prevalence | `concept_count`, `ranking`, `tie_break` | `features/selection.py`, stage 6 | per-fold selections/hashes |
| Measurement 50 x 6 | `features.measurements` | `features/construction.py` | 300-count feature manifest |
| Medication 50 x 2 + 4 | `features.medications` | construction | 104-count feature manifest |
| Procedure 50 x 2 + 3 | `features.procedures` | construction | 103-count feature manifest |
| Eight matrices | `models.yaml: matrices` | `features/matrices.py` | matrix manifest |
| Four frozen models | `models.yaml: models` | `modeling/runner.py`, stage 7 | model manifest |
| Training-only preprocessing | model configuration | `build_pipeline` per outer fold | fit manifests/tests |
| One held-out probability/combo | folds/matrices/models | pipeline stage 7 | restricted OOF rows |
| Fold and pooled performance | threshold 0.5 | `evaluation/analysis.py` | fold, summary, pooled CSVs |
| Patient bootstrap CIs | repetitions/seed/level | `evaluation/bootstrap.py` | confidence interval CSV |
| Model selection | selection hierarchy/order | `_select_models` | best model CSV |
| Full ROC and 90% specificity | target 0.90 | `evaluation/metrics.py` | ROC and operating-point CSVs |
| Highest-risk 10% | `top_risk_fraction` | `top_risk_analysis` | top-risk CSV |
| Calibration and ECE | `calibration_bins` | calibration functions | summary/coordinate CSVs |
| Decision curves | 0.01-0.50 thresholds | `_decision_curve` | decision coordinate CSV |
| Paired domain increments | comparison list | `_paired_comparisons` | paired-comparison CSV |
| Performance table | selected-model outputs | stage 8 | performance table |
| Clinical-utility table | operating/top-risk joins | stage 8 | clinical-utility table |
| Calibration table | calibration summary | stage 8 | calibration table |
| Figure inputs | evaluation thresholds/bins | stage 8 only | ROC/calibration/decision CSVs; no plots |
| Reproducibility provenance | all configs | audit/pipeline | dataset/fold/domain/matrix/model/run manifests |
