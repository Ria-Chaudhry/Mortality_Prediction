from __future__ import annotations


def categorical_columns(columns: list[str], numeric_columns: list[str]) -> list[str]:
    numeric = set(numeric_columns)
    return [column for column in columns if column not in numeric]
