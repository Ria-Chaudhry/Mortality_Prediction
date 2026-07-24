from __future__ import annotations

import pandas as pd


def aggregate_events(
    events: pd.DataFrame,
    value_col: str = "value",
    stats: tuple[str, ...] = ("mean", "min", "max", "count", "last"),
) -> pd.DataFrame:
    """Aggregate long-form events into encounter-level domain feature columns."""
    required = {"encounter_id", "domain", "feature_name", value_col}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing event columns for aggregation: {sorted(missing)}")
    if events.empty:
        return pd.DataFrame(columns=["encounter_id"])

    frame = events.copy()
    if "event_time" in frame.columns:
        frame["event_time"] = pd.to_datetime(frame["event_time"])
        frame = frame.sort_values(["encounter_id", "domain", "feature_name", "event_time"])

    grouped = frame.groupby(["encounter_id", "domain", "feature_name"], dropna=False)[value_col]
    available_stats = [stat for stat in stats if stat in {"mean", "min", "max", "count", "last"}]
    aggregated = grouped.agg(available_stats).reset_index()
    long = aggregated.melt(
        id_vars=["encounter_id", "domain", "feature_name"],
        var_name="stat",
        value_name="feature_value",
    )
    long["matrix_column"] = (
        long["domain"].astype(str)
        + "__"
        + long["feature_name"].astype(str)
        + "__"
        + long["stat"].astype(str)
    )
    wide = long.pivot_table(
        index="encounter_id",
        columns="matrix_column",
        values="feature_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide
