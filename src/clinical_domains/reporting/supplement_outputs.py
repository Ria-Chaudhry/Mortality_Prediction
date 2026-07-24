from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_supplement_predictions(predictions: pd.DataFrame, output_dir: str | Path) -> Path:
    target = Path(output_dir) / "supplement_predictions.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(target, index=False)
    return target
