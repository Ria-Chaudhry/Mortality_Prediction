"""Fail-closed artifact classification and disclosure checks."""

from __future__ import annotations

import re
from pathlib import Path

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
        r"(?i)(?:/Users/[^/\s]+|/home/[^/\s]+|/mnt/[^/\s]+|[A-Z]:\\Users\\[^\\\s]+)"
    ),
    "connection detail": re.compile(
        r"(?i)(?:host|server|database|dsn)\s*[:=]\s*(?!<|example|unset)[^\s,;]{3,}"
    ),
}

DISALLOWED_PUBLIC_COLUMNS = re.compile(
    r"(?i)(?:(^|_)(patient|person|subject|visit|admission|encounter)"
    r"(_id|_identifier|_number)?$|(date|datetime|timestamp)$|"
    r"^(probability|prediction|feature_value|raw_value)$)"
)

PUBLIC_AGGREGATE_ALLOWLIST = {
    "all_metric_confidence_intervals.csv",
    "attrition.csv",
    "calibration_coordinates.csv",
    "event_linkage_audit.csv",
    "feature_manifest.csv",
    "fold_metric_summaries.csv",
    "fold_metrics.csv",
    "fold_summary.csv",
    "matrix_manifest.csv",
    "measurement_unit_audit.csv",
    "model_selection_results.csv",
    "pooled_oof_metrics.csv",
    "prespecified_paired_matrix_comparisons.csv",
    "selected_model_calibration_table.csv",
    "selected_model_clinical_utility_table.csv",
    "selected_model_performance_table.csv",
    "selected_models_calibration_coordinates.csv",
    "selected_models_calibration_summary.csv",
    "selected_models_decision_curve_coordinates.csv",
    "selected_models_roc_coordinates.csv",
    "selected_models_sensitivity_at_90_specificity.csv",
    "selected_models_top_10_percent_risk_analysis.csv",
}


def scan_public_tree(
    root: Path,
    *,
    classification: str = "public_synthetic",
    small_cell_threshold: int | None = None,
    release_approved: bool = False,
) -> None:
    """Scan public/release-candidate artifacts without logging cell values."""
    if classification not in {
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
    findings = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() in {".pdf", ".parquet", ".gz"}:
            continue
        relative = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}: {label}")
        if path.suffix.lower() != ".csv":
            continue
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            findings.append(f"{relative}: unreadable CSV")
            continue
        disallowed = [
            str(column)
            for column in frame
            if DISALLOWED_PUBLIC_COLUMNS.search(str(column))
        ]
        if disallowed:
            findings.append(f"{relative}: disallowed public columns {sorted(disallowed)}")
        if classification == "public_clinical":
            if path.name not in PUBLIC_AGGREGATE_ALLOWLIST:
                findings.append(f"{relative}: file is absent from public schema allowlist")
            findings.extend(
                _small_cell_findings(relative, frame, int(small_cell_threshold))
            )
    if findings:
        raise IntegrityError(f"Public output privacy scan failed: {findings}")


def _small_cell_findings(
    relative: str, frame: pd.DataFrame, threshold: int
) -> list[str]:
    findings: list[str] = []
    count_columns = [
        column
        for column in frame
        if re.search(
            r"(?i)(^|_)(n|count|visits|patients|events|deaths|tp|tn|fp|fn|flagged)$",
            str(column),
        )
    ]
    for column in count_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if ((numeric > 0) & (numeric < threshold)).any():
            findings.append(f"{relative}: small cell in approved count column {column}")
    return findings
