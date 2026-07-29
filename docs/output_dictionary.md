# Output dictionary

| Output | Content |
|---|---|
| `attrition.csv` | Counts after each cohort rule; written before a count-mismatch failure. |
| `expected_vs_observed_counts.csv` | Final cohort and configured count targets against observed values, absolute differences, and tolerances. |
| `expected_vs_observed_attrition_counts.csv` | Every configured cohort attrition stage compared with observed visit/patient counts. |
| `expected_vs_observed_event_counts.csv` | Every configured domain/linkage/window stage compared with observed counts; persisted before failure. |
| `expected_vs_observed_selection_counts.csv` | Per-fold/domain selected-concept, candidate-column, and selected-column counts; persisted before failure. |
| `failed_run_manifest.json` | Safe restricted diagnostic status for a failed count check. |
| `fold_summary.csv` | Aggregate visit, patient, and outcome counts per fold. |
| `event_linkage_audit.csv` | Direct, bridge, patient-time, unmatched, and window counts. |
| `event_linkage_restricted.csv` | Restricted event-level linkage strategy/reason evidence; never public for clinical runs. |
| `fold_concept_selections.csv` | Per-fold top-50 rank/prevalence/semantics/units/hash when classification permits. |
| `fold_derived_feature_selections.csv` | Every 300/104/103 candidate row with source concept, summary, pre-imputation training support, score, construction-order tie-break, rank, selected flag, versioned rule, and hash. Exactly 21 are selected per fold/domain. Restricted for clinical runs. |
| `feature_manifest.csv` | Constructed/retained counts plus schema and value hashes. |
| `matrix_manifest.csv` | Matrix row/count, schema hash, and feature-matrix value hash. |
| `fold_metrics.csv`, `fold_metric_summaries.csv` | Fold estimates and mean/sample SD. |
| `pooled_oof_metrics.csv` | All 32 pooled combinations and threshold-0.5 metrics. |
| `all_metric_confidence_intervals.csv` | Patient-clustered percentile intervals/invalid replicates. |
| `best_model_by_matrix.csv` | Frozen AUPRC/AUROC/Brier/model-order selection. |
| `selected_models_roc_coordinates.csv` | Full selected-model ROC coordinates. |
| `selected_models_sensitivity_at_90_specificity.csv` | Selected operating point, achieved specificity, and PPV. |
| `selected_models_top_10_percent_risk_analysis.csv` | Exact ceiling-size risk group and deterministic ties. |
| `selected_models_calibration_summary.csv` | Brier, intercept/slope, calibration-in-the-large, ECE. |
| `selected_models_calibration_coordinates.csv` | Quantile-bin coordinates and event-rate intervals. |
| `selected_models_decision_curve_coordinates.csv` | Model, treat-all, treat-none net benefit and intervals. |
| `prespecified_paired_matrix_comparisons.csv` | Shared-patient-bootstrap matrix increments. |
| `shap_fold_aggregates.csv` | Restricted clinical (public synthetic) fold-level mean absolute held-out permutation SHAP; never encounter-level values. |
| `shap_summary.csv` | Safe cross-fold selected-model SHAP aggregates and ranks. |
| selected-model tables | Performance, clinical utility, and calibration manuscript inputs. |
| `manifests/*.json` | Input/config/mapping/cohort/row/fold/selection/domain/matrix/model/output provenance, including exact training/validation/preprocessing-fit partition hashes and a fitted preprocessing-state hash. |

Restricted outputs include source identities, frozen cohort/baseline, assignments, prepared
events, fold features, clinical concept lists, feature importance, and OOF predictions.

`feature_schema_hash` covers ordered names/types. `feature_value_hash` and
`feature_matrix_hash` cover identities, ordered columns/types, and values. Canonical analytical
input signatures are row-order-independent where source row order is incidental.

Synthetic verification performs exact same-run file checksum validation and separately pins every
deterministic aggregate using a cross-platform canonical table contract (ordered columns/rows and
numeric values serialized at ten decimal places). Timestamp, Git-state, and raw output-byte hashes
are excluded from stable run-manifest equality. Missing, unexpected, or mutated artifacts fail
verification.
