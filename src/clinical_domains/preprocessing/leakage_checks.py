from __future__ import annotations


def assert_no_outcome_columns(feature_columns: list[str]) -> None:
    forbidden_tokens = ("outcome", "mortality", "death", "died")
    offending = [
        column for column in feature_columns if any(token in column.lower() for token in forbidden_tokens)
    ]
    if offending:
        raise ValueError(f"Potential outcome leakage columns detected: {offending}")
