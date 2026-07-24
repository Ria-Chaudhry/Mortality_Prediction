from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    target = ensure_parent(path)
    if target.suffix == ".csv":
        frame.to_csv(target, index=False)
        return
    if target.suffix == ".parquet":
        frame.to_parquet(target, index=False)
        return
    raise ValueError(f"Unsupported table format: {target.suffix}")
