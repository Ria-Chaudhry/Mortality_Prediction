from __future__ import annotations

import pandas as pd


def require_minimum_age(encounters: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    if "age" not in encounters.columns:
        return encounters.copy()
    return encounters.loc[encounters["age"].fillna(-1) >= min_age].reset_index(drop=True)


def require_valid_time_order(encounters: pd.DataFrame) -> pd.DataFrame:
    frame = encounters.copy()
    frame["admit_time"] = pd.to_datetime(frame["admit_time"])
    frame["discharge_time"] = pd.to_datetime(frame["discharge_time"])
    return frame.loc[frame["discharge_time"] >= frame["admit_time"]].reset_index(drop=True)
