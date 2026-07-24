from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pandas as pd


def validate_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> None:
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_records_against_schema(frame: pd.DataFrame, schema_path: str | Path) -> None:
    """Validate dataframe rows against a JSON schema for standardized row records."""
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    for record in frame.to_dict(orient="records"):
        jsonschema.validate(record, schema)
