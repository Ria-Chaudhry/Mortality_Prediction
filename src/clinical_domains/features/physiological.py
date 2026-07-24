from __future__ import annotations

import pandas as pd


def physiological_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[events["domain"] == "physiological"].reset_index(drop=True)
