from __future__ import annotations

import pandas as pd


def restrict_events_to_landmark(
    events: pd.DataFrame,
    encounters: pd.DataFrame,
    hours: float = 24,
    event_time_col: str = "event_time",
    admit_time_col: str = "admit_time",
) -> pd.DataFrame:
    """Keep events occurring from admission through the configured landmark window."""
    required_events = {"encounter_id", event_time_col}
    required_encounters = {"encounter_id", admit_time_col}
    missing = (required_events - set(events.columns)) | (required_encounters - set(encounters.columns))
    if missing:
        raise ValueError(f"Missing required landmark columns: {sorted(missing)}")

    event_frame = events.copy()
    encounter_frame = encounters[["encounter_id", admit_time_col]].copy()
    event_frame[event_time_col] = pd.to_datetime(event_frame[event_time_col])
    encounter_frame[admit_time_col] = pd.to_datetime(encounter_frame[admit_time_col])

    merged = event_frame.merge(encounter_frame, on="encounter_id", how="inner")
    merged["hours_from_admit"] = (
        merged[event_time_col] - merged[admit_time_col]
    ).dt.total_seconds() / 3600.0

    in_window = merged["hours_from_admit"].between(0, hours, inclusive="both")
    return merged.loc[in_window].drop(columns=[admit_time_col]).reset_index(drop=True)
