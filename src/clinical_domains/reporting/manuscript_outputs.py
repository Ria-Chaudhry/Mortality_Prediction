from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_manuscript_tables(metrics: pd.DataFrame, output_dir: str | Path) -> Path:
    target = Path(output_dir) / "manuscript_metrics.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(target, index=False)
    return target
