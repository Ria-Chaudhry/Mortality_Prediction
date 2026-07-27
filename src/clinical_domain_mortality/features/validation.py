"""Hard-fail leakage and feature-integrity checks."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..errors import IntegrityError, LeakageError


def assert_no_forbidden_features(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    if not frame.index.is_unique:
        raise IntegrityError("Feature matrix row key is not unique")
    if len(frame.columns) != len(set(frame.columns)):
        raise IntegrityError("Feature matrix has duplicate column names")
    patterns = config["features"]["forbidden_name_patterns"]
    violations = []
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).casefold())
        if any(re.search(rf"(^|_){re.escape(pattern)}(_|$)", normalized) for pattern in patterns):
            violations.append(str(column))
    if violations:
        raise LeakageError(f"Forbidden predictors detected: {sorted(violations)}")
    prohibited_types = [
        column
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
        or pd.api.types.is_timedelta64_dtype(frame[column])
    ]
    if prohibited_types:
        raise LeakageError(f"Temporal predictors detected: {prohibited_types}")
