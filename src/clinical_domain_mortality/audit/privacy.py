"""Fail-closed artifact classification and disclosure checks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import IntegrityError

SENSITIVE_PATTERNS = {
    "credential assignment": re.compile(
        r"(?i)(password|passwd|api[_-]?key|access[_-]?token|secret)\s*[:=]\s*[^\s'\"]{4,}"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "database URL with authority": re.compile(
        r"(?i)(?:postgres(?:ql)?|mysql|mssql|oracle)://[^/\s]+@"
    ),
    "internal absolute path": re.compile(
        r"(?i)(?:/Users/[^/\s]+|/home/[^/\s]+|/mnt/[^/\s]+|"
        r"/(?:Volumes|private|srv|opt)/[^/\s]+|[A-Z]:\\Users\\[^\\\s]+)"
    ),
    "connection detail": re.compile(
        r"(?i)(?:host|server|database|dsn)\s*[:=]\s*(?!<|example|unset)[^\s,;]{3,}"
    ),
}

DISALLOWED_PUBLIC_COLUMNS = re.compile(
    r"(?i)(?:(^|_)(patient|person|subject|visit|admission|encounter|member|case|record)"
    r"(_id|_identifier|_number|_key)?$|(^|_)(mrn|hadm_id|subject_id)$|"
    r"(date|datetime|timestamp|timepoint)$|"
    r"^(probability|prediction|oof_probability|shap_value|feature_value|raw_value|"
    r"fold_assignment)$)"
)

# Public clinical artifacts are exact-schema products. Unknown files and columns
# fail closed. Selection rows, unit audits, fold assignments, OOF predictions,
# event rows, and feature values are intentionally absent.
PUBLIC_CLINICAL_TABLE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "all_metric_confidence_intervals.csv": (
        "matrix", "model", "metric", "estimate", "ci_lower", "ci_upper",
        "confidence_level", "bootstrap_repetitions", "valid_replicates",
        "invalid_replicates", "bootstrap_unit", "ci_method",
    ),
    "attrition.csv": ("step", "visits", "patients"),
    "best_model_by_matrix.csv": (
        "matrix", "model", "auprc", "auroc", "brier", "selection_rule",
    ),
    "event_linkage_audit.csv": ("domain", "status", "count"),
    "expected_vs_observed_counts.csv": (
        "category", "stage", "measure", "expected", "observed",
        "absolute_difference", "tolerance", "matches",
    ),
    "expected_vs_observed_attrition_counts.csv": (
        "category", "stage", "measure", "expected", "observed",
        "absolute_difference", "tolerance", "matches",
    ),
    "expected_vs_observed_event_counts.csv": (
        "domain", "attrition_stage", "expected", "observed",
        "absolute_difference", "tolerance", "matches",
    ),
    "expected_vs_observed_selection_counts.csv": (
        "fold", "domain", "measure", "expected", "observed",
        "absolute_difference", "tolerance", "matches",
    ),
    "feature_manifest.csv": (
        "fold", "domain", "feature_count", "constructed_feature_count",
        "feature_schema_hash", "feature_value_hash", "derived_selection_hash",
        "selection_hash",
    ),
    "fold_metric_summaries.csv": (
        "matrix", "model", "metric", "fold_mean", "fold_sd", "valid_folds",
    ),
    "fold_metrics.csv": (
        "matrix", "model", "fold", "visits", "events", "auroc", "auprc",
        "brier", "log_loss", "threshold", "tn", "fp", "fn", "tp", "accuracy",
        "balanced_accuracy", "ppv", "npv", "sensitivity", "specificity", "f1",
    ),
    "fold_summary.csv": ("fold", "visits", "patients", "events"),
    "matrix_manifest.csv": (
        "fold", "matrix", "rows", "input_feature_count", "feature_schema_hash",
        "feature_matrix_hash",
    ),
    "pooled_oof_metrics.csv": (
        "matrix", "model", "visits", "events", "auroc", "auprc", "brier",
        "log_loss", "threshold", "tn", "fp", "fn", "tp", "accuracy",
        "balanced_accuracy", "ppv", "npv", "sensitivity", "specificity", "f1",
    ),
    "prespecified_paired_matrix_comparisons.csv": (
        "comparator_matrix", "comparator_model", "expanded_matrix", "expanded_model",
        "metric", "difference", "difference_definition", "ci_lower", "ci_upper",
        "valid_replicates", "invalid_replicates", "shared_patient_bootstrap",
    ),
    "selected_model_calibration_table.csv": (
        "matrix", "model", "brier", "calibration_intercept", "calibration_slope",
        "calibration_in_the_large", "mean_predicted_risk", "observed_prevalence",
        "expected_calibration_error", "requested_bin_count", "actual_bin_count",
    ),
    "selected_model_clinical_utility_table.csv": (
        "matrix", "model", "specificity_target", "sensitivity", "specificity",
        "threshold", "ppv_at_90_specificity", "flagged_count_at_90_specificity",
        "flagged_per_100_at_90_specificity", "requested_fraction",
        "flagged_count_top_10_percent", "flagged_per_100_top_10_percent",
        "deaths_captured", "total_deaths", "death_capture_fraction",
        "ppv_top_10_percent", "prevalence", "enrichment", "cutoff_probability",
        "cutoff_ties_total", "cutoff_ties_flagged",
    ),
    "selected_model_performance_table.csv": (
        "matrix", "model", "visits", "events", "auroc", "auprc", "brier",
        "log_loss", "threshold", "tn", "fp", "fn", "tp", "accuracy",
        "balanced_accuracy", "ppv", "npv", "sensitivity", "specificity", "f1",
        "auroc_ci_lower", "auroc_ci_upper", "auprc_ci_lower", "auprc_ci_upper",
        "brier_ci_lower", "brier_ci_upper",
    ),
    "selected_models_calibration_coordinates.csv": (
        "matrix", "model", "bin", "count", "events", "mean_predicted_risk",
        "observed_event_rate", "minimum_probability", "maximum_probability",
        "event_rate_ci_lower", "event_rate_ci_upper",
    ),
    "selected_models_calibration_summary.csv": (
        "matrix", "model", "brier", "calibration_intercept", "calibration_slope",
        "calibration_in_the_large", "mean_predicted_risk", "observed_prevalence",
        "expected_calibration_error", "requested_bin_count", "actual_bin_count",
    ),
    "selected_models_decision_curve_coordinates.csv": (
        "matrix", "model", "threshold", "strategy", "net_benefit", "ci_lower",
        "ci_upper", "valid_replicates", "invalid_replicates",
    ),
    "selected_models_roc_coordinates.csv": (
        "matrix", "model", "threshold", "false_positive_rate",
        "true_positive_rate", "specificity", "sensitivity",
    ),
    "selected_models_sensitivity_at_90_specificity.csv": (
        "matrix", "model", "specificity_target", "sensitivity", "specificity",
        "threshold", "ppv", "flagged_count", "flagged_per_100",
    ),
    "selected_models_top_10_percent_risk_analysis.csv": (
        "matrix", "model", "requested_fraction", "flagged_count", "flagged_per_100",
        "deaths_captured", "total_deaths", "death_capture_fraction", "ppv",
        "prevalence", "enrichment", "cutoff_probability", "cutoff_ties_total",
        "cutoff_ties_flagged",
    ),
    "shap_summary.csv": (
        "dataset", "feature_matrix", "model", "feature", "clinical_domain",
        "source_concept", "summary_type", "explainer", "mean_absolute_shap",
        "folds", "mean_background_rows", "mean_evaluation_rows", "rank",
        "fold_aggregation_policy",
    ),
}

PUBLIC_CLINICAL_JSON_SCHEMAS: dict[str, frozenset[str] | None] = {
    "cohort_manifest.json": frozenset({
        "cohort_hash", "row_order_hash", "visits", "patients", "outcomes",
        "landmark_hours", "predictor_window_hours", "outcome_horizon_days",
        "death_time_precision_counts", "death_source_conflict_count",
    }),
    "mapping_validation.json": frozenset({
        "mapping_hash", "domains", "mimic_native_rules",
    }),
    "source_mapping_validation.json": frozenset({
        "mapping_confirmed", "mapping_hash", "adapter", "table_schema_hashes",
        "table_row_counts",
    }),
    "run_manifest.json": frozenset({
        "run_id", "dataset", "adapter", "created_utc", "git_commit",
        "git_worktree_dirty", "code_hash", "config_hash", "mapping_hash",
        "cohort_hash", "row_order_hash", "fold_hash", "input_hashes",
        "output_hashes", "software_versions", "warnings", "failures",
        "artifact_classification", "privacy_gate", "paper_reproduction_status",
        "restricted_artifacts",
    }),
    "manifests/configuration_manifest.json": frozenset({
        "config_hash", "mapping_hash", "cohort_hash", "row_order_hash", "fold_hash",
    }),
    "manifests/dataset_manifest.json": frozenset({
        "dataset", "visits", "patients", "events", "folds", "matrices", "models",
        "model_fits", "oof_probabilities_per_visit", "retained_features_per_domain",
        "bootstrap_repetitions",
    }),
    "manifests/domain_manifest.json": frozenset({"domains", "expected_counts"}),
    "manifests/fold_manifest.json": frozenset({
        "fold_hash", "selection_hashes", "folds",
    }),
    "manifests/input_manifest.json": frozenset({
        "dataset", "adapter", "input_hashes", "source_release_or_snapshot",
        "config_hash", "mapping_hash", "input_collection_hash",
    }),
    "manifests/mapping_manifest.json": frozenset({
        "mapping_confirmed", "mapping_hash", "adapter", "input_table_count",
    }),
    "manifests/matrix_manifest.json": frozenset({
        "matrix_count", "definitions", "fold_matrices",
    }),
    "manifests/model_manifest.json": frozenset({
        "fit_count", "models", "matrices", "fits",
    }),
    "manifests/output_manifest.json": None,
    "manifests/selection_manifest.json": frozenset({
        "selection_count", "selection_hashes", "ranking", "tie_break",
        "derived_feature_selection_count", "derived_selection_hashes",
        "derived_feature_ranking", "derived_feature_tie_break",
        "retained_per_domain_per_fold", "training_folds_only",
        "combined_audit_rows",
    }),
}

PUBLIC_SMALL_CELL_COLUMNS: dict[str, tuple[str, ...]] = {
    "attrition.csv": ("visits", "patients"),
    "event_linkage_audit.csv": ("count",),
    "expected_vs_observed_counts.csv": ("expected", "observed"),
    "expected_vs_observed_attrition_counts.csv": ("expected", "observed"),
    "expected_vs_observed_event_counts.csv": ("expected", "observed"),
    "fold_metrics.csv": ("visits", "events", "tn", "fp", "fn", "tp"),
    "fold_summary.csv": ("visits", "patients", "events"),
    "matrix_manifest.csv": ("rows",),
    "pooled_oof_metrics.csv": ("visits", "events", "tn", "fp", "fn", "tp"),
    "selected_model_clinical_utility_table.csv": (
        "flagged_count_at_90_specificity", "flagged_count_top_10_percent",
        "deaths_captured", "total_deaths", "cutoff_ties_total",
        "cutoff_ties_flagged",
    ),
    "selected_model_performance_table.csv": (
        "visits", "events", "tn", "fp", "fn", "tp",
    ),
    "selected_models_calibration_coordinates.csv": ("count", "events"),
    "selected_models_sensitivity_at_90_specificity.csv": ("flagged_count",),
    "selected_models_top_10_percent_risk_analysis.csv": (
        "flagged_count", "deaths_captured", "total_deaths",
        "cutoff_ties_total", "cutoff_ties_flagged",
    ),
}

SAFE_PUBLIC_JSON_FIELDS = {
    "oof_probabilities_per_visit",
}
SAFE_PUBLIC_JSON_CONTAINERS = {
    "death_time_precision_counts",
}


def scan_public_tree(
    root: Path,
    *,
    classification: str = "public_synthetic",
    small_cell_threshold: int | None = None,
    release_approved: bool = False,
) -> None:
    """Scan all proposed artifacts without logging protected cell values."""
    if classification not in {
        "restricted",
        "public_synthetic",
        "release_candidate_aggregate",
        "public_clinical",
    }:
        raise IntegrityError(f"Unknown artifact classification: {classification}")
    if classification == "public_clinical":
        if not release_approved:
            raise IntegrityError("Clinical publication gate lacks explicit release approval")
        if small_cell_threshold is None or int(small_cell_threshold) < 1:
            raise IntegrityError(
                "Clinical publication gate requires an approved small-cell threshold"
            )

    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        suffixes = [suffix.lower() for suffix in path.suffixes]
        is_csv = path.suffix.lower() == ".csv" or suffixes[-2:] == [".csv", ".gz"]
        is_parquet = path.suffix.lower() == ".parquet"
        is_json = path.suffix.lower() == ".json"

        if classification == "public_clinical":
            allowed = (
                relative in PUBLIC_CLINICAL_TABLE_SCHEMAS
                or relative in PUBLIC_CLINICAL_JSON_SCHEMAS
            )
            if not allowed:
                findings.append(f"{relative}: file is absent from public schema allowlist")

        if not is_parquet:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                text = ""
            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(text):
                    findings.append(f"{relative}: {label}")

        if is_csv or is_parquet:
            frame = _read_public_table(path, relative, findings)
            if frame is None:
                continue
            if classification != "restricted":
                findings.extend(_disallowed_column_findings(relative, frame))
            if classification == "public_clinical" and relative in PUBLIC_CLINICAL_TABLE_SCHEMAS:
                expected = list(PUBLIC_CLINICAL_TABLE_SCHEMAS[relative])
                if list(map(str, frame.columns)) != expected:
                    findings.append(
                        f"{relative}: public column schema differs from allowlist"
                    )
                findings.extend(
                    _small_cell_findings(
                        relative,
                        frame,
                        int(small_cell_threshold),
                        PUBLIC_SMALL_CELL_COLUMNS.get(relative, ()),
                    )
                )
        elif is_json:
            payload = _read_public_json(path, relative, findings)
            if payload is None:
                continue
            if classification != "restricted":
                findings.extend(_json_identifier_findings(relative, payload))
            if classification == "public_clinical" and relative in PUBLIC_CLINICAL_JSON_SCHEMAS:
                schema = PUBLIC_CLINICAL_JSON_SCHEMAS[relative]
                if not isinstance(payload, dict):
                    findings.append(f"{relative}: public JSON must be an object")
                elif schema is not None and set(payload) != set(schema):
                    findings.append(
                        f"{relative}: public JSON field schema differs from allowlist"
                    )
                findings.extend(
                    _json_small_cell_findings(
                        relative, payload, int(small_cell_threshold)
                    )
                )
        elif classification == "public_clinical":
            findings.append(f"{relative}: unsupported public artifact format")

    if findings:
        raise IntegrityError(f"Public output privacy scan failed: {sorted(set(findings))}")


def _read_public_table(
    path: Path, relative: str, findings: list[str]
) -> pd.DataFrame | None:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
        findings.append(f"{relative}: unreadable tabular artifact")
        return None


def _read_public_json(
    path: Path, relative: str, findings: list[str]
) -> Any | None:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        findings.append(f"{relative}: unreadable JSON artifact")
        return None


def _disallowed_column_findings(relative: str, frame: pd.DataFrame) -> list[str]:
    disallowed = [
        str(column) for column in frame if DISALLOWED_PUBLIC_COLUMNS.search(str(column))
    ]
    return (
        [f"{relative}: disallowed public columns {sorted(disallowed)}"]
        if disallowed
        else []
    )


def _json_identifier_findings(relative: str, payload: Any) -> list[str]:
    findings: list[str] = []
    for path, value in _walk_json(payload):
        key = path.rsplit(".", 1)[-1]
        components = set(path.replace("[", ".").replace("]", "").split("."))
        aggregate_container = bool(components & SAFE_PUBLIC_JSON_CONTAINERS)
        if (
            key not in SAFE_PUBLIC_JSON_FIELDS
            and not aggregate_container
            and DISALLOWED_PUBLIC_COLUMNS.search(key)
        ):
            findings.append(f"{relative}: disallowed public JSON field {path}")
        if isinstance(value, str):
            for label, pattern in SENSITIVE_PATTERNS.items():
                if pattern.search(value):
                    findings.append(f"{relative}: {label} in JSON field {path}")
    return findings


def _small_cell_findings(
    relative: str,
    frame: pd.DataFrame,
    threshold: int,
    count_columns: tuple[str, ...],
) -> list[str]:
    findings: list[str] = []
    for column in count_columns:
        if column not in frame:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if ((numeric > 0) & (numeric < threshold)).any():
            findings.append(f"{relative}: small cell in approved count column {column}")
    return findings


def _json_small_cell_findings(
    relative: str, payload: Any, threshold: int
) -> list[str]:
    findings: list[str] = []
    for path, value in _walk_json(payload):
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        key = path.rsplit(".", 1)[-1].casefold()
        parent = path.rsplit(".", 1)[0].casefold() if "." in path else ""
        count_semantic = (
            key in {
                "visits", "patients", "events", "outcomes", "deaths",
                "death_source_conflict_count", "input_table_count",
                "training_rows", "validation_rows", "qualifying_events",
                "rows",
            }
            or parent.endswith("row_counts")
            or key.endswith("_rows")
            or key.endswith("_events")
            or key.endswith("_patient_count")
            or key.endswith("_visit_count")
            or key.endswith("_event_count")
            or key.endswith("_death_count")
            or key in {"tn", "fp", "fn", "tp"}
        )
        if count_semantic and 0 < float(value) < threshold:
            findings.append(f"{relative}: small cell in approved JSON count field {path}")
    return findings


def _walk_json(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.append((path, item))
            rows.extend(_walk_json(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            rows.extend(_walk_json(item, path))
    return rows
