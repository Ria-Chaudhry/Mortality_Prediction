from __future__ import annotations

import pandas as pd


def dataframe_audit(frame: pd.DataFrame) -> dict[str, object]:
    """Return small, non-identifying audit metadata for a dataframe."""
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }
