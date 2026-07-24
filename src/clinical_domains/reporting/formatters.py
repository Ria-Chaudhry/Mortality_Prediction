from __future__ import annotations


def format_metric(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"
