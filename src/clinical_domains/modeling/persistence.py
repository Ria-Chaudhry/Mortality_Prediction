from __future__ import annotations

import pickle
from pathlib import Path


def save_model(model, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str | Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
