"""Field-level schema validation for source-neutral tables."""

from __future__ import annotations

import pandas as pd

from ..errors import SchemaError

STANDARD_COLUMNS = {
    "patients": ["patient_id", "birth_datetime", "sex", "race", "ethnicity"],
    "encounters": [
        "visit_id",
        "patient_id",
        "start_datetime",
        "end_datetime",
        "visit_type",
        "elective",
        "followup_end_datetime",
    ],
    "deaths": ["patient_id", "death_datetime"],
    "diagnoses": [
        "diagnosis_id",
        "visit_id",
        "patient_id",
        "diagnosis_datetime",
        "code",
        "source_table",
    ],
    "bridge": ["bridge_key", "visit_id"],
}

EVENT_COLUMNS = [
    "event_id",
    "source_visit_id",
    "bridge_key",
    "patient_id",
    "event_datetime",
    "concept_key",
    "concept_name",
    "value",
    "unit",
    "source_table",
    "semantics",
]


def validate_standardized(tables: dict[str, pd.DataFrame]) -> None:
    required_tables = {
        "patients",
        "encounters",
        "deaths",
        "diagnoses",
        "measurements",
        "medications",
        "procedures",
        "bridge",
        "metadata",
    }
    missing_tables = required_tables - set(tables)
    if missing_tables:
        raise SchemaError(f"Missing standardized tables: {sorted(missing_tables)}")
    for name, columns in STANDARD_COLUMNS.items():
        _require_columns(tables[name], name, columns)
    for name in ("measurements", "medications", "procedures"):
        _require_columns(tables[name], name, EVENT_COLUMNS)
    _require_columns(
        tables["metadata"],
        "metadata",
        ["domain", "concept_key", "concept_name", "source_table", "semantics", "unit"],
    )
    _require_unique(tables["patients"], "patients", ["patient_id"])
    _require_unique(tables["encounters"], "encounters", ["visit_id"])
    for name in ("measurements", "medications", "procedures"):
        _require_unique(tables[name], name, ["event_id"])


def _require_columns(frame: pd.DataFrame, name: str, columns: list[str]) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise SchemaError(f"{name} missing required columns: {sorted(missing)}")


def _require_unique(frame: pd.DataFrame, name: str, columns: list[str]) -> None:
    if not frame.empty and frame.duplicated(columns, keep=False).any():
        raise SchemaError(f"{name} contains duplicate keys: {columns}")
