# Output dictionary

| Output | Content |
|---|---|
| `attrition.csv` | Visit/patient counts after every cohort rule. |
| `fold_summary.csv` | Aggregate visit, patient, and outcome counts per frozen fold. |
| `event_linkage_audit.csv` | Direct, bridge, patient-time, unmatched, window-excluded counts. |
| `fold_concept_selections.csv` | Synthetic/non-sensitive fold ranks, prevalence, semantics, units, hashes. |
| `feature_manifest.csv` | Fold/domain feature counts and hashes. |
| `feature_dictionary.csv` | Synthetic/non-sensitive fold/domain feature names and selection hashes. |
| `fold_metrics.csv` | All 160 validation-fold metric rows. |
| `fold_metric_summaries.csv` | Mean and sample SD across five folds. |
| `pooled_oof_metrics.csv` | Performance and threshold-0.5 metrics for all 32 combinations. |
| `all_metric_confidence_intervals.csv` | Patient-percentile bootstrap intervals and invalid counts. |
| `best_model_by_matrix.csv` | AUPRC/AUROC/Brier/frozen-order selection per matrix. |
| `selected_model_performance_table.csv` | Manuscript-ready performance estimates and intervals. |
| `selected_model_clinical_utility_table.csv` | 90%-specificity and top-10%-risk results. |
| `selected_model_calibration_table.csv` | Brier, intercept/slope, calibration-in-large, ECE, bins. |
| `selected_models_roc_coordinates.csv` | Complete, non-interpolated selected-model ROC points. |
| `selected_models_calibration_coordinates.csv` | Deterministic quantile-bin counts, risks, rates, Wilson CIs. |
| `selected_models_decision_curve_coordinates.csv` | Model/treat-all/treat-none net benefit and bootstrap CIs. |
| `prespecified_paired_matrix_comparisons.csv` | Shared-patient-bootstrap matrix increments. |
| `manifests/*.json` | Input/output/config/mapping/cohort/row/fold/selection/software provenance. |

Restricted outputs contain `base_acute_care_cohort.parquet`, `baseline_X.parquet`,
`fold_assignments_restricted.csv`, prepared event Parquets, protected selection/importance files
when needed, all per-fold 300/104/103 domain matrices, and `oof_predictions_restricted.csv`.

Analytical CSVs retain full precision. Plotting code must read coordinates without recomputing
statistics.
