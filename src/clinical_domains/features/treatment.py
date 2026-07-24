from __future__ import annotations

import pandas as pd


def treatment_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[events["domain"] == "treatment"].reset_index(drop=True)
