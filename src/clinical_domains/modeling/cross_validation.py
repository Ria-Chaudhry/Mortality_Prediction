from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold


def grouped_cv_splits(
    frame: pd.DataFrame,
    group_col: str = "patient_id",
    target_col: str = "outcome",
    n_splits: int = 5,
    seed: int = 20260724,
):
    """Yield patient-grouped train/test indices, stratified when feasible."""
    if group_col not in frame.columns:
        raise ValueError(f"Missing group column: {group_col}")
    groups = frame[group_col].to_numpy()

    unique_groups = np.unique(groups)
    effective_splits = min(n_splits, len(unique_groups))
    if effective_splits < 2:
        raise ValueError("At least two patient groups are required for grouped CV")

    if target_col in frame.columns and frame[target_col].nunique(dropna=True) > 1:
        splitter = StratifiedGroupKFold(
            n_splits=effective_splits, shuffle=True, random_state=seed
        )
        yield from splitter.split(frame, frame[target_col], groups)
    else:
        splitter = GroupKFold(n_splits=effective_splits)
        yield from splitter.split(frame, groups=groups)
