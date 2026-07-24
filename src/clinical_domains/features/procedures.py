from __future__ import annotations

import pandas as pd


def procedure_events(events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[events["domain"] == "procedures"].reset_index(drop=True)
